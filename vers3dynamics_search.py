#!/usr/bin/env python3
"""
Vers3Dynamics Search - RF Spectrum Situational Awareness System

A real-time 3D visualization platform for monitoring electromagnetic spectrum activity,
detecting patterns, anomalies, and temporal stability in the RF environment.

Author: Vers3Dynamics
License: MIT
"""

import numpy as np
import plotly.graph_objects as go
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
from threading import Thread, Lock
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System configuration parameters"""
    # Frequency spectrum
    FREQ_START = 88e6      # 88 MHz
    FREQ_END = 2.4e9       # 2.4 GHz
    NUM_BANDS = 64         # Number of frequency bands to monitor
    
    # Temporal parameters
    SAMPLE_RATE = 10       # Samples per second
    WINDOW_SIZE = 100      # Historical samples for baseline calculation
    UPDATE_INTERVAL = 0.1  # Seconds between updates
    
    # Anomaly detection
    ANOMALY_THRESHOLD = 2.5    # Standard deviations for anomaly detection
    STABILITY_THRESHOLD = 0.15  # Variance threshold for stability assessment
    
    # Visualization
    PORT = 5000
    AUTO_ROTATE = True
    POINT_SIZE = 8
    
    # Simulation (when no real SDR available)
    SIMULATE_SDR = True
    NUM_SIGNAL_SOURCES = 8


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpectralNode:
    """Represents a frequency band in the spectral field"""
    frequency: float        # Center frequency (Hz)
    power: float           # Current power level (dBm)
    baseline: float        # Historical baseline power
    variance: float        # Temporal variance
    anomaly_score: float   # Deviation from baseline
    stability: float       # Temporal stability metric (0-1)
    band_class: str       # "low", "mid", "high" frequency classification
    position: Tuple[float, float, float]  # 3D coordinates
    history: deque         # Recent power measurements


@dataclass
class SignalSource:
    """Simulated RF signal source"""
    frequency: float       # Center frequency
    bandwidth: float       # Signal bandwidth
    power: float          # Base power level
    modulation: str       # Type of modulation
    duty_cycle: float     # Transmission duty cycle
    phase: float         # Current phase


# =============================================================================
# RF SIGNAL PROCESSING
# =============================================================================

class RFProcessor:
    """Processes RF spectrum data and computes spectral features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.bands = self._initialize_bands()
        self.lock = Lock()
        
    def _initialize_bands(self) -> List[SpectralNode]:
        """Initialize frequency band nodes"""
        bands = []
        freq_range = self.config.FREQ_END - self.config.FREQ_START
        freq_step = freq_range / self.config.NUM_BANDS
        
        for i in range(self.config.NUM_BANDS):
            freq = self.config.FREQ_START + (i + 0.5) * freq_step
            
            # Classify band
            normalized_freq = i / self.config.NUM_BANDS
            if normalized_freq < 0.33:
                band_class = "low"
            elif normalized_freq < 0.67:
                band_class = "mid"
            else:
                band_class = "high"
            
            # Calculate 3D position (cylindrical arrangement)
            angle = 2 * np.pi * i / self.config.NUM_BANDS
            radius = 10 + 5 * normalized_freq
            height = normalized_freq * 20 - 10
            
            position = (
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            )
            
            node = SpectralNode(
                frequency=freq,
                power=-90.0,  # Initial noise floor
                baseline=-90.0,
                variance=0.0,
                anomaly_score=0.0,
                stability=1.0,
                band_class=band_class,
                position=position,
                history=deque(maxlen=self.config.WINDOW_SIZE)
            )
            bands.append(node)
        
        return bands
    
    def update_spectrum(self, power_data: np.ndarray):
        """Update spectral nodes with new power measurements"""
        with self.lock:
            for i, node in enumerate(self.bands):
                if i < len(power_data):
                    power = power_data[i]
                    node.history.append(power)
                    node.power = power
                    
                    # Compute statistics
                    if len(node.history) >= 10:
                        history_array = np.array(node.history)
                        node.baseline = np.median(history_array)
                        node.variance = np.var(history_array)
                        
                        # Anomaly score (z-score)
                        std = np.sqrt(node.variance) if node.variance > 0 else 1.0
                        node.anomaly_score = abs(power - node.baseline) / std
                        
                        # Stability metric (inverse of normalized variance)
                        normalized_var = node.variance / (abs(node.baseline) + 1e-6)
                        node.stability = max(0, 1 - normalized_var / self.config.STABILITY_THRESHOLD)
    
    def get_spectral_state(self) -> Dict:
        """Get current spectral state for visualization"""
        with self.lock:
            nodes_data = []
            powers = []
            stabilities = []
            for node in self.bands:
                powers.append(node.power)
                stabilities.append(node.stability)
                nodes_data.append({
                    'frequency': node.frequency,
                    'power': node.power,
                    'baseline': node.baseline,
                    'anomaly_score': node.anomaly_score,
                    'stability': node.stability,
                    'band_class': node.band_class,
                    'position': node.position
                })
            
            num_anomalies = sum(1 for n in self.bands if n.anomaly_score > self.config.ANOMALY_THRESHOLD)
            strongest_node = max(self.bands, key=lambda n: n.power)
            average_power = float(np.mean(powers)) if powers else -90.0
            average_stability = float(np.mean(stabilities)) if stabilities else 1.0

            return {
                'nodes': nodes_data,
                'timestamp': datetime.now().isoformat(),
                'num_anomalies': num_anomalies,
                'summary': {
                    'average_power': average_power,
                    'average_stability': average_stability,
                    'strongest_frequency': strongest_node.frequency,
                    'strongest_power': strongest_node.power,
                    'anomaly_ratio': num_anomalies / self.config.NUM_BANDS
                }
            }


# =============================================================================
# SDR INTERFACE / SIMULATOR
# =============================================================================

class SpectrumSource:
    """Provides RF spectrum data - simulated or from real SDR"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sources = self._initialize_sources()
        self.time = 0
        
    def _initialize_sources(self) -> List[SignalSource]:
        """Initialize simulated signal sources"""
        sources = []
        freq_range = self.config.FREQ_END - self.config.FREQ_START
        
        for i in range(self.config.NUM_SIGNAL_SOURCES):
            freq = self.config.FREQ_START + np.random.random() * freq_range
            sources.append(SignalSource(
                frequency=freq,
                bandwidth=1e6 * (0.5 + np.random.random() * 2),  # 0.5-2.5 MHz
                power=-60 + np.random.random() * 30,  # -60 to -30 dBm
                modulation=np.random.choice(['FM', 'AM', 'QAM', 'PSK']),
                duty_cycle=0.3 + np.random.random() * 0.7,
                phase=np.random.random() * 2 * np.pi
            ))
        
        return sources
    
    def get_spectrum_snapshot(self) -> np.ndarray:
        """Get current spectrum power across all bands"""
        freq_range = self.config.FREQ_END - self.config.FREQ_START
        freq_step = freq_range / self.config.NUM_BANDS
        
        # Initialize with noise floor
        spectrum = np.random.normal(-90, 2, self.config.NUM_BANDS)
        
        # Add signal sources
        for source in self.sources:
            # Determine which bands are affected
            source_band = int((source.frequency - self.config.FREQ_START) / freq_step)
            bandwidth_bands = int(source.bandwidth / freq_step) + 1
            
            for offset in range(-bandwidth_bands, bandwidth_bands + 1):
                band_idx = source_band + offset
                if 0 <= band_idx < self.config.NUM_BANDS:
                    # Signal envelope
                    distance = abs(offset) / bandwidth_bands
                    envelope = np.exp(-distance * 3)
                    
                    # Temporal modulation
                    active = np.sin(source.phase + self.time) * 0.5 + 0.5 > (1 - source.duty_cycle)
                    
                    if active:
                        signal_power = source.power * envelope + np.random.normal(0, 1)
                        spectrum[band_idx] = 10 * np.log10(
                            10**(spectrum[band_idx]/10) + 10**(signal_power/10)
                        )
            
            # Update source phase
            source.phase += 0.1 + np.random.random() * 0.2
        
        # Add occasional anomalies
        if np.random.random() < 0.05:
            anomaly_band = np.random.randint(0, self.config.NUM_BANDS)
            spectrum[anomaly_band] += 20 + np.random.random() * 15
        
        self.time += 0.1
        return spectrum


# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

class Visualizer:
    """Generates 3D visualization of spectral field"""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_hold_power = np.full(self.config.NUM_BANDS, -np.inf)
        
    def create_3d_scene(self, spectral_state: Dict) -> str:
        """Create Plotly 3D visualization"""
        nodes = spectral_state['nodes']
        num_nodes = len(nodes)

        # Terrain coordinates: frequency on X, power on Y, stability-derived depth on Z
        freq_mhz = np.array([n['frequency'] / 1e6 for n in nodes])
        powers = np.array([n['power'] for n in nodes])
        stabilities = np.array([n['stability'] for n in nodes])
        z_depth = stabilities * 25.0 - 12.5

        # Keep a max-hold history for persistent transient visibility
        if len(self.max_hold_power) != num_nodes:
            self.max_hold_power = np.full(num_nodes, -np.inf)
        self.max_hold_power = np.maximum(self.max_hold_power, powers)

        # Marker sizes scale with signal power intensity
        power_normalized = (powers - powers.min()) / (powers.max() - powers.min() + 1e-6)
        sizes = 5 + power_normalized * 14
        
        # Create hover text
        hover_texts = []
        for node in nodes:
            node_freq_mhz = node['frequency'] / 1e6
            text = (f"Freq: {node_freq_mhz:.1f} MHz<br>"
                   f"Power: {node['power']:.1f} dBm<br>"
                   f"Baseline: {node['baseline']:.1f} dBm<br>"
                   f"Anomaly: {node['anomaly_score']:.2f}σ<br>"
                   f"Stability: {node['stability']:.2%}<br>"
                   f"Class: {node['band_class']}")
            hover_texts.append(text)
        
        # Main spectral wireframe trace
        nodes_trace = go.Scatter3d(
            x=freq_mhz,
            y=powers,
            z=z_depth,
            mode='lines+markers',
            line=dict(
                color='rgba(80, 230, 255, 0.55)',
                width=4
            ),
            marker=dict(
                size=sizes,
                color=powers,
                colorscale='Turbo',
                cmin=-95,
                cmax=-20,
                opacity=0.95,
                line=dict(color='rgba(170, 245, 255, 0.9)', width=0.7),
                colorbar=dict(
                    title='Power (dBm)',
                    thickness=12,
                    len=0.68,
                    x=1.02,
                    y=0.55,
                    outlinecolor='rgba(0, 255, 255, 0.5)',
                    tickfont=dict(color='rgba(184, 242, 255, 0.9)')
                )
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Spectral Terrain'
        )

        # Persistent max-hold ghost trace
        max_hold_trace = go.Scatter3d(
            x=freq_mhz,
            y=self.max_hold_power,
            z=np.full(num_nodes, 13.5),
            mode='lines',
            line=dict(color='rgba(255, 85, 255, 0.45)', width=3, dash='dot'),
            hovertemplate='Max Hold<br>%{x:.1f} MHz<br>%{y:.1f} dBm<extra></extra>',
            name='Max Hold Ghost'
        )

        # Semi-transparent reference plane at noise floor
        x_plane = np.array([[freq_mhz.min(), freq_mhz.max()], [freq_mhz.min(), freq_mhz.max()]])
        y_plane = np.full((2, 2), -90.0)
        z_plane = np.array([[-14.0, -14.0], [14.0, 14.0]])
        noise_floor_plane = go.Surface(
            x=x_plane,
            y=y_plane,
            z=z_plane,
            showscale=False,
            opacity=0.2,
            colorscale=[[0, 'rgba(0, 100, 140, 0.3)'], [1, 'rgba(0, 210, 255, 0.3)']],
            hoverinfo='skip',
            name='Noise Floor (-90 dBm)'
        )

        # Bloom/halo overlay for high anomaly nodes
        anomaly_nodes = [n for n in nodes if n['anomaly_score'] > self.config.ANOMALY_THRESHOLD]
        anomaly_bloom_trace = go.Scatter3d(
            x=[n['frequency'] / 1e6 for n in anomaly_nodes],
            y=[n['power'] for n in anomaly_nodes],
            z=[(n['stability'] * 25.0 - 12.5) for n in anomaly_nodes],
            mode='markers',
            marker=dict(
                size=26,
                color='rgba(255, 30, 140, 0.25)',
                line=dict(color='rgba(255, 90, 180, 0.6)', width=1.2)
            ),
            hoverinfo='skip',
            name='Anomaly Bloom'
        )
        
        # Observer origin
        observer_trace = go.Scatter3d(
            x=[float(np.mean(freq_mhz))], y=[-88], z=[-11],
            mode='markers',
            marker=dict(
                size=15,
                color='yellow',
                symbol='diamond',
                line=dict(color='orange', width=2)
            ),
            text=['Observer Platform'],
            hoverinfo='text',
            name='Observer'
        )
        
        # Create figure
        fig = go.Figure(data=[noise_floor_plane, max_hold_trace, nodes_trace, anomaly_bloom_trace, observer_trace])
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f"Vers3Dynamics Search - Spectral Situational Awareness<br>"
                     f"<sub>Timestamp: {spectral_state['timestamp']} | "
                     f"Anomalies: {spectral_state['num_anomalies']}</sub>",
                x=0.5,
                xanchor='center',
                font=dict(color='rgba(130, 240, 255, 0.95)', size=20)
            ),
            scene=dict(
                xaxis=dict(
                    title='Frequency (MHz)',
                    backgroundcolor='rgb(7, 10, 26)',
                    gridcolor='rgba(48, 109, 158, 0.45)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 255, 255, 0.25)'
                ),
                yaxis=dict(
                    title='Power (dBm)',
                    range=[-100, -20],
                    backgroundcolor='rgb(7, 10, 26)',
                    gridcolor='rgba(48, 109, 158, 0.45)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 255, 255, 0.25)'
                ),
                zaxis=dict(
                    title='Stability Field',
                    range=[-15, 15],
                    backgroundcolor='rgb(7, 10, 26)',
                    gridcolor='rgba(48, 109, 158, 0.45)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 255, 255, 0.25)'
                ),
                bgcolor='rgb(3, 6, 17)',
                camera=dict(
                    eye=dict(x=1.7, y=1.25, z=0.9)
                )
            ),
            paper_bgcolor='rgb(2, 4, 14)',
            font=dict(color='rgb(200, 244, 255)', family='Orbitron, Share Tech Mono, monospace'),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(2, 10, 28, 0.75)',
                bordercolor='rgba(0, 255, 255, 0.45)',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig.to_json()


# =============================================================================
# CORE SYSTEM
# =============================================================================

class Vers3DynamicsSearch:
    """Main system coordinator"""
    
    def __init__(self):
        self.config = Config()
        self.processor = RFProcessor(self.config)
        self.source = SpectrumSource(self.config)
        self.visualizer = Visualizer(self.config)
        self.running = False
        self.acquisition_thread = None
        self.started_at = None
        
        logger.info("Vers3Dynamics Search initialized")
        logger.info(f"Monitoring: {self.config.FREQ_START/1e6:.1f} - {self.config.FREQ_END/1e9:.3f} GHz")
        logger.info(f"Bands: {self.config.NUM_BANDS} | Update rate: {self.config.SAMPLE_RATE} Hz")
    
    def start_acquisition(self):
        """Start RF spectrum acquisition"""
        self.running = True
        self.started_at = datetime.now()
        self.acquisition_thread = Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()
        logger.info("Spectrum acquisition started")
    
    def stop_acquisition(self):
        """Stop RF spectrum acquisition"""
        self.running = False
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=2)
        logger.info("Spectrum acquisition stopped")
    
    def _acquisition_loop(self):
        """Continuous acquisition and processing loop"""
        while self.running:
            try:
                # Get spectrum snapshot
                spectrum = self.source.get_spectrum_snapshot()
                
                # Process spectrum
                self.processor.update_spectrum(spectrum)
                
                # Sleep until next sample
                time.sleep(1.0 / self.config.SAMPLE_RATE)
                
            except Exception as e:
                logger.error(f"Acquisition error: {e}")
                time.sleep(0.5)
    
    def get_visualization_data(self) -> str:
        """Get current visualization data as JSON"""
        spectral_state = self.processor.get_spectral_state()
        return self.visualizer.create_3d_scene(spectral_state)

    def get_status_snapshot(self) -> Dict:
        """Return current runtime status metrics for the API/UI."""
        uptime_seconds = 0.0
        if self.started_at:
            uptime_seconds = (datetime.now() - self.started_at).total_seconds()

        return {
            'running': self.running,
            'bands': self.config.NUM_BANDS,
            'freq_range': f"{self.config.FREQ_START/1e6:.1f} - {self.config.FREQ_END/1e9:.3f} GHz",
            'sample_rate_hz': self.config.SAMPLE_RATE,
            'uptime_seconds': round(uptime_seconds, 1)
        }


# =============================================================================
# WEB INTERFACE
# =============================================================================

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vers3Dynamics Search - RF Spectrum Monitor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --hud-cyan: #61f4ff;
            --hud-blue: #1d76ff;
            --hud-bg: #02040d;
            --hud-panel: rgba(2, 13, 34, 0.78);
            --hud-border: rgba(97, 244, 255, 0.72);
        }
        
        body {
            font-family: 'Orbitron', 'Share Tech Mono', monospace;
            background: radial-gradient(circle at 20% 20%, #041536 0%, #020611 40%, #010309 100%);
            color: var(--hud-cyan);
            overflow: hidden;
        }

        body::after {
            content: '';
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 20;
            background: repeating-linear-gradient(
                to bottom,
                rgba(150, 240, 255, 0.04) 0px,
                rgba(150, 240, 255, 0.04) 1px,
                rgba(0, 0, 0, 0.0) 2px,
                rgba(0, 0, 0, 0.0) 4px
            );
            mix-blend-mode: screen;
        }
        
        .header {
            background: linear-gradient(90deg, rgba(0, 7, 22, 0.95), rgba(2, 14, 40, 0.85));
            padding: 15px 30px;
            border-bottom: 1px solid var(--hud-border);
            box-shadow: 0 0 22px rgba(30, 180, 255, 0.25);
            display: flex;
            justify-content: space-between;
            align-items: center;
            letter-spacing: 0.08em;
        }
        
        .header h1 {
            font-size: 24px;
            text-shadow: 0 0 12px rgba(97, 244, 255, 0.7);
        }
        
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
            font-family: 'Share Tech Mono', monospace;
        }

        .status strong {
            color: var(--hud-cyan);
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff9d;
            box-shadow: 0 0 10px #00ff9d;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            position: relative;
            height: calc(100vh - 80px);
            padding: 20px;
            z-index: 1;
        }
        
        #plot {
            width: 100%;
            height: 100%;
            border: 1px solid var(--hud-border);
            clip-path: polygon(0 18px, 18px 0, calc(100% - 18px) 0, 100% 18px, 100% calc(100% - 18px), calc(100% - 18px) 100%, 18px 100%, 0 calc(100% - 18px));
            background: rgba(1, 8, 20, 0.7);
            box-shadow: inset 0 0 22px rgba(40, 170, 255, 0.22), 0 0 26px rgba(0, 185, 255, 0.15);
        }

        .hud-panel {
            position: absolute;
            background: var(--hud-panel);
            border: 1px solid var(--hud-border);
            padding: 15px;
            color: #bcf7ff;
            clip-path: polygon(0 14px, 14px 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%);
            box-shadow: 0 0 18px rgba(60, 205, 255, 0.28), inset 0 0 24px rgba(25, 105, 180, 0.25);
            backdrop-filter: blur(1px);
        }
        
        .legend-panel {
            bottom: 30px;
            right: 30px;
            min-width: 280px;
        }

        .metrics-panel {
            bottom: 30px;
            left: 30px;
            min-width: 310px;
            font-size: 12px;
            line-height: 1.6;
        }

        .metrics-panel h3,
        .legend-panel h3 {
            margin-bottom: 10px;
            font-size: 14px;
            border-bottom: 1px solid rgba(97, 244, 255, 0.45);
            padding-bottom: 5px;
            letter-spacing: 0.15em;
        }

        .metrics-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            font-family: 'Share Tech Mono', monospace;
        }

        .metrics-value {
            color: #8de0ff;
            font-weight: bold;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            font-size: 12px;
            font-family: 'Share Tech Mono', monospace;
        }
        
        .color-box {
            width: 20px;
            height: 20px;
            border: 1px solid rgba(190, 248, 255, 0.75);
            box-shadow: 0 0 10px rgba(88, 208, 255, 0.4);
        }
        
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            text-align: center;
        }
        
        .spinner {
            border: 4px solid rgba(0, 255, 255, 0.1);
            border-top: 4px solid #00ffff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Vers3Dynamics Search</h1>
        <div class="status">
            <div class="status-item">
                <div class="indicator"></div>
                <span>ACTIVE</span>
            </div>
            <div class="status-item">
                <span id="update-counter">Updates: 0</span>
            </div>
            <div class="status-item">
                <span>Anomalies: <strong id="anomaly-counter">0</strong></span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div id="plot"></div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Initializing Spectral Analysis...</div>
        </div>
    </div>
    
    <div class="legend-panel hud-panel">
        <h3>LEGEND</h3>
        <div class="legend-item">
            <div class="color-box" style="background: linear-gradient(90deg, #2c1a92, #00dcff, #ffe600);"></div>
            <span>Continuous Power Map (Turbo)</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background: rgba(255, 85, 255, 0.65);"></div>
            <span>Max-Hold Ghost Trace</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background: rgba(0, 180, 230, 0.45);"></div>
            <span>Noise Floor Reference Plane (-90 dBm)</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background: rgba(255, 35, 150, 0.35);"></div>
            <span>⚠ Anomaly Bloom Halo</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background: yellow;"></div>
            <span>◆ Observer Platform</span>
        </div>
    </div>

    <div class="metrics-panel hud-panel">
        <h3>LIVE METRICS</h3>
        <div class="metrics-row"><span>Avg Power</span><span class="metrics-value" id="avg-power">-</span></div>
        <div class="metrics-row"><span>Avg Stability</span><span class="metrics-value" id="avg-stability">-</span></div>
        <div class="metrics-row"><span>Strongest Band</span><span class="metrics-value" id="strongest-band">-</span></div>
        <div class="metrics-row"><span>Anomaly Ratio</span><span class="metrics-value" id="anomaly-ratio">-</span></div>
        <div class="metrics-row"><span>Last Update</span><span class="metrics-value" id="last-update">-</span></div>
    </div>
    
    <script>
        let updateCount = 0;
        
        function updateVisualization() {
            fetch('/api/spectrum')
                .then(response => response.json())
                .then(data => {
                    const plotData = JSON.parse(data.plot);
                    const layout = plotData.layout;
                    const plotDiv = document.getElementById('plot');
                    
                    // Hide loading on first update
                    if (updateCount === 0) {
                        document.getElementById('loading').style.display = 'none';
                    }
                    
                    // Update or create plot
                    if (updateCount === 0) {
                        Plotly.newPlot(plotDiv, plotData.data, layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false
                        });
                    } else {
                        Plotly.react(plotDiv, plotData.data, layout);
                    }
                    
                    updateCount++;
                    document.getElementById('update-counter').textContent = 
                        `Updates: ${updateCount}`;

                    if (data.summary) {
                        document.getElementById('avg-power').textContent = `${data.summary.average_power.toFixed(1)} dBm`;
                        document.getElementById('avg-stability').textContent = `${(data.summary.average_stability * 100).toFixed(1)}%`;
                        document.getElementById('strongest-band').textContent = 
                            `${(data.summary.strongest_frequency / 1e6).toFixed(1)} MHz @ ${data.summary.strongest_power.toFixed(1)} dBm`;
                        document.getElementById('anomaly-ratio').textContent = `${(data.summary.anomaly_ratio * 100).toFixed(1)}%`;
                    }

                    if (data.timestamp) {
                        const localTime = new Date(data.timestamp).toLocaleTimeString();
                        document.getElementById('last-update').textContent = localTime;
                    }

                    document.getElementById('anomaly-counter').textContent = data.num_anomalies ?? 0;
                })
                .catch(error => {
                    console.error('Error updating visualization:', error);
                    document.getElementById('last-update').textContent = 'Connection issue';
                });
        }
        
        // Initial update
        updateVisualization();
        
        // Auto-update every 500ms
        setInterval(updateVisualization, 500);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            Plotly.Plots.resize(document.getElementById('plot'));
        });
    </script>
</body>
</html>
"""

# Flask application
app = Flask(__name__)
CORS(app)
system = None


def ensure_system_initialized() -> Vers3DynamicsSearch:
    """Lazily initialize the runtime system for WSGI/serverless environments."""
    global system
    if system is None:
        system = Vers3DynamicsSearch()
        system.start_acquisition()
    return system

@app.route('/')
def index():
    """Serve main interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/spectrum')
def get_spectrum():
    """API endpoint for spectrum data"""
    try:
        active_system = ensure_system_initialized()
        spectral_state = active_system.processor.get_spectral_state()
        plot_json = active_system.visualizer.create_3d_scene(spectral_state)
        return jsonify({
            'plot': plot_json,
            'status': 'active',
            'num_anomalies': spectral_state['num_anomalies'],
            'summary': spectral_state['summary'],
            'timestamp': spectral_state['timestamp']
        })
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    active_system = ensure_system_initialized()
    return jsonify(active_system.get_status_snapshot())


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    global system
    
    print("=" * 70)
    print("  Vers3Dynamics Search - RF Spectrum Situational Awareness System")
    print("=" * 70)
    print()
    print("  Initializing electromagnetic environment monitoring...")
    print()
    
    active_system = ensure_system_initialized()
    
    print(f"  ✓ Spectrum acquisition active")
    print(f"  ✓ Monitoring {active_system.config.NUM_BANDS} frequency bands")
    print(f"  ✓ Web interface starting on http://localhost:{active_system.config.PORT}")
    print()
    print("  Opening browser interface...")
    print("  Press Ctrl+C to stop")
    print()
    print("=" * 70)
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=active_system.config.PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        active_system.stop_acquisition()
        print("Vers3Dynamics Search terminated.")

if __name__ == '__main__':
    main()
