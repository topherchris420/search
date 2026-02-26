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
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
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

    @staticmethod
    def _read_env_int(name: str, default: int, minimum: int = None) -> int:
        """Read an integer from env with validation and fallback."""
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        try:
            parsed = int(raw_value)
        except ValueError:
            logger.warning("Invalid integer for %s=%r; using default %s", name, raw_value, default)
            return default

        if minimum is not None and parsed < minimum:
            logger.warning("%s=%s is below minimum %s; using minimum", name, parsed, minimum)
            return minimum

        return parsed

    @staticmethod
    def _read_env_float(name: str, default: float, minimum: float = None) -> float:
        """Read a float from env with validation and fallback."""
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        try:
            parsed = float(raw_value)
        except ValueError:
            logger.warning("Invalid float for %s=%r; using default %s", name, raw_value, default)
            return default

        if minimum is not None and parsed < minimum:
            logger.warning("%s=%s is below minimum %s; using minimum", name, parsed, minimum)
            return minimum

        return parsed

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables for deployment flexibility."""
        config = cls()
        config.PORT = cls._read_env_int("RF_MONITOR_PORT", config.PORT, minimum=1)
        config.SAMPLE_RATE = cls._read_env_int("RF_SAMPLE_RATE_HZ", config.SAMPLE_RATE, minimum=1)
        config.ANOMALY_THRESHOLD = cls._read_env_float(
            "RF_ANOMALY_SIGMA",
            config.ANOMALY_THRESHOLD,
            minimum=0.0
        )
        return config


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
    pattern: str         # "emerging", "fading", "stable"
    trend: float         # temporal trend slope
    position: Tuple[float, float, float]  # 3D coordinates


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
        
        # Vectorized history buffer: (NUM_BANDS, WINDOW_SIZE)
        self.history_buffer = np.full((self.config.NUM_BANDS, self.config.WINDOW_SIZE), -90.0)
        self.history_idx = 0
        self.samples_processed = 0

        # Precompute trend weights for N=25
        # x = [0, 1/(N-1), ..., 1]
        # x_prime = x - mean(x)
        # weights = x_prime / sum(x_prime^2)
        N = 25
        x = np.linspace(0.0, 1.0, N)
        x_prime = x - np.mean(x)
        self.trend_weights = x_prime / np.sum(x_prime**2)

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
                pattern="stable",
                trend=0.0,
                position=position
            )
            bands.append(node)
        
        return bands
    
    def update_spectrum(self, power_data: np.ndarray):
        """Update spectral nodes with new power measurements"""
        with self.lock:
            # Update history buffer
            self.history_buffer[:, self.history_idx] = power_data
            self.history_idx = (self.history_idx + 1) % self.config.WINDOW_SIZE
            self.samples_processed += 1

            if self.samples_processed < 10:
                # Just update current power
                for i, node in enumerate(self.bands):
                     if i < len(power_data):
                         node.power = power_data[i]
                return

            # Vectorized statistics
            # Note: We compute over the entire window
            baselines = np.median(self.history_buffer, axis=1)
            variances = np.var(self.history_buffer, axis=1)

            # Vectorized Trend (last 25 samples)
            # Get indices for the last 25 samples in chronological order
            indices = (np.arange(self.history_idx - 25, self.history_idx) + self.config.WINDOW_SIZE) % self.config.WINDOW_SIZE
            recent_history = self.history_buffer[:, indices]
            # Matrix multiplication for slopes: (NUM_BANDS, 25) @ (25,) -> (NUM_BANDS,)
            trends = np.dot(recent_history, self.trend_weights)

            # Vectorized anomaly scores
            stds = np.sqrt(variances)
            stds[stds == 0] = 1.0
            anomaly_scores = np.abs(power_data - baselines) / stds

            # Vectorized stability
            normalized_vars = variances / (np.abs(baselines) + 1e-6)
            stabilities = np.maximum(0.0, 1.0 - normalized_vars / self.config.STABILITY_THRESHOLD)

            # Update nodes
            # Still need to iterate to update object fields, but heavy math is done
            for i, node in enumerate(self.bands):
                if i < len(power_data):
                    node.power = power_data[i]
                    node.baseline = baselines[i]
                    node.variance = variances[i]
                    node.anomaly_score = anomaly_scores[i]
                    node.stability = stabilities[i]
                    node.trend = trends[i]
                    
                    if node.trend > 0.08:
                        node.pattern = "emerging"
                    elif node.trend < -0.08:
                        node.pattern = "fading"
                    else:
                        node.pattern = "stable"
    
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
                    'pattern': node.pattern,
                    'trend': node.trend,
                    'band_class': node.band_class,
                    'position': node.position
                })
            
            num_anomalies = sum(1 for n in self.bands if n.anomaly_score > self.config.ANOMALY_THRESHOLD)
            strongest_node = max(self.bands, key=lambda n: n.power)
            average_power = float(np.mean(powers)) if powers else -90.0
            average_stability = float(np.mean(stabilities)) if stabilities else 1.0
            pattern_counts = {
                'emerging': sum(1 for n in self.bands if n.pattern == 'emerging'),
                'fading': sum(1 for n in self.bands if n.pattern == 'fading'),
                'stable': sum(1 for n in self.bands if n.pattern == 'stable')
            }

            return {
                'nodes': nodes_data,
                'timestamp': datetime.now().isoformat(),
                'num_anomalies': num_anomalies,
                'summary': {
                    'average_power': average_power,
                    'average_stability': average_stability,
                    'strongest_frequency': strongest_node.frequency,
                    'strongest_power': strongest_node.power,
                    'anomaly_ratio': num_anomalies / self.config.NUM_BANDS,
                    'pattern_counts': pattern_counts
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
        
    def create_3d_scene(self, spectral_state: Dict) -> str:
        """Create Plotly 3D visualization"""
        nodes = spectral_state['nodes']
        
        # Extract data for visualization
        x = [n['position'][0] for n in nodes]
        y = [n['position'][1] for n in nodes]
        z = [n['position'][2] for n in nodes]
        
        # Size based on power (normalized)
        powers = np.array([n['power'] for n in nodes])
        power_normalized = (powers - powers.min()) / (powers.max() - powers.min() + 1e-6)
        sizes = 5 + power_normalized * 20
        
        # Color and opacity based on anomaly score, class, and stability
        colors = []
        for node in nodes:
            stability_alpha = max(0.2, min(1.0, node['stability'] * 0.8 + 0.2))
            if node['anomaly_score'] > self.config.ANOMALY_THRESHOLD:
                base_rgb = (255, 0, 0)  # anomaly
            elif node['band_class'] == 'low':
                base_rgb = (0, 255, 255)
            elif node['band_class'] == 'mid':
                base_rgb = (50, 255, 50)
            else:
                base_rgb = (255, 0, 255)
            colors.append(f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {stability_alpha:.3f})")
        
        # Create hover text
        hover_texts = []
        for node in nodes:
            freq_mhz = node['frequency'] / 1e6
            text = (f"Freq: {freq_mhz:.1f} MHz<br>"
                   f"Power: {node['power']:.1f} dBm<br>"
                   f"Baseline: {node['baseline']:.1f} dBm<br>"
                   f"Anomaly: {node['anomaly_score']:.2f}σ<br>"
                   f"Stability: {node['stability']:.2%}<br>"
                   f"Pattern: {node['pattern']} ({node['trend']:+.2f})<br>"
                   f"Class: {node['band_class']}")
            hover_texts.append(text)
        
        # Create spectral nodes trace
        nodes_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color='white', width=0.5)
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Spectral Nodes'
        )
        
        # Observer origin
        observer_trace = go.Scatter3d(
            x=[0], y=[0], z=[0],
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
        fig = go.Figure(data=[nodes_trace, observer_trace])
        
        # Layout
        fig.update_layout(
            title=dict(
                text="",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='X Axis', backgroundcolor='rgb(2, 10, 8)',
                          gridcolor='rgba(0, 255, 180, 0.22)', showbackground=True, zeroline=False),
                yaxis=dict(title='Y Axis', backgroundcolor='rgb(2, 10, 8)',
                          gridcolor='rgba(0, 255, 180, 0.22)', showbackground=True, zeroline=False),
                zaxis=dict(title='Frequency Axis', backgroundcolor='rgb(2, 10, 8)',
                          gridcolor='rgba(0, 255, 180, 0.22)', showbackground=True, zeroline=False),
                bgcolor='rgb(1, 8, 6)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            paper_bgcolor='rgb(0, 6, 5)',
            font=dict(color='rgb(130, 255, 210)', family='Courier New'),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0, 16, 14, 0.75)',
                bordercolor='rgba(0, 255, 200, 0.5)',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        
        return fig.to_json()


# =============================================================================
# CORE SYSTEM
# =============================================================================

class Vers3DynamicsSearch:
    """Main system coordinator"""
    
    def __init__(self):
        self.config = Config.from_env()
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

    def get_architecture_snapshot(self) -> Dict:
        """Describe orchestrated service architecture and active operating profile."""
        return {
            'system': 'Vers3Dynamics Search',
            'pipeline': [
                'RF acquisition (SDR/simulator)',
                'Band aggregation (64 channels)',
                'Baseline/anomaly analytics',
                'Pattern recognition (emerging/fading/stable)',
                '3D observer-centric visualization'
            ],
            'operating_profile': {
                'frequency_range_hz': [self.config.FREQ_START, self.config.FREQ_END],
                'monitored_bands': self.config.NUM_BANDS,
                'temporal_window_samples': self.config.WINDOW_SIZE,
                'anomaly_sigma_threshold': self.config.ANOMALY_THRESHOLD
            }
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
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --neon-primary: #00ff88;
            --neon-cyan: #00ffd5;
            --neon-magenta: #a000ff;
            --hud-bg: rgba(0, 12, 8, 0.72);
            --hud-border: rgba(0, 255, 170, 0.45);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Courier New', monospace;
            color: var(--neon-cyan);
            background:
                radial-gradient(circle at 20% 10%, rgba(0, 120, 70, 0.25), transparent 45%),
                radial-gradient(circle at 80% 80%, rgba(0, 70, 100, 0.18), transparent 55%),
                #030807;
            overflow: hidden;
            min-height: 100vh;
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background: repeating-linear-gradient(
                to bottom,
                rgba(30, 255, 180, 0.03),
                rgba(30, 255, 180, 0.03) 1px,
                transparent 2px,
                transparent 4px
            );
            animation: scanline 5s linear infinite;
            opacity: 0.35;
        }

        @keyframes scanline {
            from { transform: translateY(-20px); }
            to { transform: translateY(20px); }
        }

        .app-shell {
            position: relative;
            height: 100vh;
            width: 100vw;
            padding: 16px;
        }

        #plot {
            width: 100%;
            height: 100%;
            border: 1px solid var(--hud-border);
            border-radius: 14px;
            background: rgba(0, 0, 0, 0.4);
            box-shadow: inset 0 0 35px rgba(0, 255, 160, 0.08), 0 0 25px rgba(0, 255, 170, 0.15);
        }

        .hud-panel {
            position: absolute;
            background: var(--hud-bg);
            border: 1px solid var(--hud-border);
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 255, 170, 0.15);
            backdrop-filter: blur(4px);
        }

        .hud-title {
            font-size: 30px;
            letter-spacing: 1px;
            color: var(--neon-primary);
            text-shadow: 0 0 12px rgba(0, 255, 140, 0.6);
            margin-bottom: 4px;
        }

        .top-bar {
            top: 24px;
            left: 24px;
            right: 24px;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }

        .subtitle { font-size: 12px; color: rgba(140, 255, 220, 0.75); }

        .status {
            display: flex;
            gap: 14px;
            align-items: center;
            font-size: 13px;
        }

        .pill {
            border: 1px solid rgba(0, 255, 170, 0.45);
            border-radius: 999px;
            padding: 5px 10px;
            background: rgba(0, 255, 120, 0.1);
            color: #a8ffd7;
        }

        .indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
            background: #00ff7b;
            box-shadow: 0 0 10px #00ff7b;
            animation: pulse 1.6s infinite;
        }

        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.45;} }

        .mapper {
            top: 110px;
            left: 24px;
            width: 330px;
            padding: 14px;
            z-index: 9;
        }

        .mapper h3 {
            color: #33ff9d;
            border: 1px solid rgba(0, 255, 150, 0.45);
            padding: 6px;
            margin-bottom: 8px;
            font-size: 30px;
        }

        .mapper .line { margin: 6px 0; font-size: 18px; color: #3be9b6; }
        .mapper .line strong { color: #71ffef; }

        .controls {
            top: 110px;
            right: 24px;
            width: 320px;
            padding: 14px;
            z-index: 9;
            font-size: 12px;
        }

        .control-row { margin: 10px 0; }
        .control-row label { display: flex; justify-content: space-between; margin-bottom: 5px; }

        input[type="range"] { width: 100%; accent-color: #18ff9f; }

        .btn {
            border: 1px solid rgba(0,255,180,.45);
            background: rgba(0, 255, 150, 0.12);
            color: #9dffd9;
            border-radius: 8px;
            padding: 7px 10px;
            cursor: pointer;
            width: 100%;
            transition: .2s ease;
            font-family: inherit;
        }

        .btn:hover { background: rgba(0,255,150,.24); box-shadow: 0 0 12px rgba(0,255,150,.25); }

        .metrics {
            left: 24px;
            right: 24px;
            bottom: 24px;
            padding: 10px;
            z-index: 10;
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 8px;
        }

        .metric-card {
            background: rgba(0, 20, 14, 0.6);
            border: 1px solid rgba(0,255,170,.25);
            border-radius: 8px;
            padding: 8px;
            min-height: 64px;
        }

        .metric-card .label { font-size: 11px; color: rgba(160, 255, 220, 0.78); }
        .metric-card .value { font-size: 15px; color: #5bffbf; margin-top: 6px; font-weight: bold; }

        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-size: 20px;
            color: #8fffd4;
            z-index: 12;
        }

        .spinner {
            border: 4px solid rgba(0, 255, 180, 0.12);
            border-top: 4px solid #00ffaa;
            border-radius: 50%;
            width: 44px;
            height: 44px;
            margin: 0 auto 14px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin { from { transform: rotate(0deg);} to {transform: rotate(360deg);} }
    </style>
</head>
<body>
    <div class="app-shell">
        <div id="plot"></div>

        <div class="hud-panel top-bar">
            <div>
                <div class="hud-title">Vers3Dynamics</div>
                <div class="subtitle">Real-time RF situational analysis and anomaly triage</div>
            </div>
            <div class="status">
                <div class="pill"><span class="indicator"></span>Tracking<strong id="anomaly-counter">0</strong> ANOMALIES</div>
                <div class="pill" id="update-counter">Updates: 0</div>
                <div class="pill" id="last-update-pill">Syncing...</div>
            </div>
        </div>

        <div class="hud-panel mapper">
            <h3>R.A.I.N. Lab</h3>
            <div class="line"><strong>A-</strong> 2.4 GHz Band</div>
            <div class="line"><strong>B-</strong> 5.0 GHz Band</div>
            <div class="line" style="color:#ff44d7;"><strong>C-</strong> 6.0 GHz Band</div>
            <br>
            <div class="line">[ CORE ] : AP Signal Strength</div>
            <div class="line">[ SPHERE ] : Est. Coverage Volume</div>
            <div class="line">[ BLUR ] : Positional Uncertainty</div>
            <div class="line">[ ORIGIN ] : Local Receiver</div>
        </div>

        <div class="hud-panel controls">
            <div class="control-row"><button id="toggle-stream" class="btn">Pause Stream</button></div>
            <div class="control-row">
                <label for="refresh-slider"><span>Refresh Rate</span><strong id="refresh-value">500 ms</strong></label>
                <input id="refresh-slider" type="range" min="200" max="1500" value="500" step="100">
            </div>
            <div class="control-row">
                <label for="glow-slider"><span>Glow Intensity</span><strong id="glow-value">65%</strong></label>
                <input id="glow-slider" type="range" min="20" max="100" value="65" step="5">
            </div>
            <div class="control-row"><button id="toggle-rotate" class="btn" aria-pressed="true">Auto-Rotate: ON</button></div>
            <div class="control-row"><button id="reset-camera" class="btn">Reset Camera</button></div>
        </div>

        <div class="hud-panel metrics">
            <div class="metric-card"><div class="label">Avg Power</div><div id="avg-power" class="value">-</div></div>
            <div class="metric-card"><div class="label">Avg Stability</div><div id="avg-stability" class="value">-</div></div>
            <div class="metric-card"><div class="label">Strongest Band</div><div id="strongest-band" class="value">-</div></div>
            <div class="metric-card"><div class="label">Anomaly Ratio</div><div id="anomaly-ratio" class="value">-</div></div>
            <div class="metric-card"><div class="label">Pattern Balance</div><div id="pattern-counts" class="value">-</div></div>
            <div class="metric-card"><div class="label">Last Update</div><div id="last-update" class="value">-</div></div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            Initializing spectral mesh...
        </div>
    </div>

    <script>
        let updateCount = 0;
        let refreshMs = 500;
        let pollingId = null;
        let streamPaused = false;
        let autoRotate = true;
        let rotationStep = 0;
        let glowIntensity = 0.65;

        function startPolling() {
            if (pollingId) clearInterval(pollingId);
            pollingId = setInterval(() => {
                if (!streamPaused) updateVisualization();
            }, refreshMs);
        }

        function applyLayoutTweaks(layout) {
            const opacity = Math.max(0.18, Math.min(0.95, glowIntensity));
            const plotBg = `rgba(0, 8, 7, ${0.75 + opacity * 0.2})`;
            layout.paper_bgcolor = plotBg;
            layout.scene.bgcolor = `rgba(0, 8, 7, ${0.65 + opacity * 0.25})`;

            if (autoRotate) {
                rotationStep += 0.035;
                layout.scene.camera = {
                    eye: {
                        x: 1.75 * Math.cos(rotationStep),
                        y: 1.75 * Math.sin(rotationStep),
                        z: 1.05 + 0.2 * Math.sin(rotationStep * 0.8)
                    }
                };
            }
        }

        function updateVisualization() {
            fetch('/api/spectrum')
                .then(response => response.json())
                .then(data => {
                    const plotData = JSON.parse(data.plot);
                    const layout = plotData.layout;
                    const plotDiv = document.getElementById('plot');

                    applyLayoutTweaks(layout);

                    if (updateCount === 0) {
                        document.getElementById('loading').style.display = 'none';
                        Plotly.newPlot(plotDiv, plotData.data, layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false
                        });
                    } else {
                        Plotly.react(plotDiv, plotData.data, layout, { responsive: true, displaylogo: false });
                    }

                    updateCount++;
                    document.getElementById('update-counter').textContent = `Updates: ${updateCount}`;

                    if (data.summary) {
                        document.getElementById('avg-power').textContent = `${data.summary.average_power.toFixed(1)} dBm`;
                        document.getElementById('avg-stability').textContent = `${(data.summary.average_stability * 100).toFixed(1)}%`;
                        document.getElementById('strongest-band').textContent = `${(data.summary.strongest_frequency / 1e6).toFixed(1)} MHz`;
                        document.getElementById('anomaly-ratio').textContent = `${(data.summary.anomaly_ratio * 100).toFixed(1)}%`;
                        if (data.summary.pattern_counts) {
                            const { emerging, fading } = data.summary.pattern_counts;
                            document.getElementById('pattern-counts').textContent = `${emerging} ↑ / ${fading} ↓`;
                        }
                    }

                    if (data.timestamp) {
                        const localTime = new Date(data.timestamp).toLocaleTimeString();
                        document.getElementById('last-update').textContent = localTime;
                        document.getElementById('last-update-pill').textContent = `SYNC ${localTime}`;
                    }

                    document.getElementById('anomaly-counter').textContent = data.num_anomalies ?? 0;
                })
                .catch(error => {
                    console.error('Error updating visualization:', error);
                    document.getElementById('last-update').textContent = 'Connection issue';
                    document.getElementById('last-update-pill').textContent = 'LINK DEGRADED';
                });
        }

        document.getElementById('toggle-stream').addEventListener('click', (event) => {
            streamPaused = !streamPaused;
            event.target.textContent = streamPaused ? 'Resume Stream' : 'Pause Stream';
            if (!streamPaused) updateVisualization();
        });

        document.getElementById('toggle-rotate').addEventListener('click', (event) => {
            autoRotate = !autoRotate;
            event.target.textContent = `Auto-Rotate: ${autoRotate ? 'ON' : 'OFF'}`;
            event.target.setAttribute('aria-pressed', autoRotate);
        });

        document.getElementById('refresh-slider').addEventListener('input', (event) => {
            refreshMs = Number(event.target.value);
            document.getElementById('refresh-value').textContent = `${refreshMs} ms`;
            startPolling();
        });

        document.getElementById('glow-slider').addEventListener('input', (event) => {
            glowIntensity = Number(event.target.value) / 100;
            document.getElementById('glow-value').textContent = `${event.target.value}%`;
        });

        document.getElementById('reset-camera').addEventListener('click', () => {
            rotationStep = 0;
            autoRotate = true;
            const btn = document.getElementById('toggle-rotate');
            btn.textContent = 'Auto-Rotate: ON';
            btn.setAttribute('aria-pressed', 'true');
            updateVisualization();
        });

        updateVisualization();
        startPolling();

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

@app.route('/api/architecture')
def get_architecture():
    """Expose orchestrated system architecture and operating profile."""
    active_system = ensure_system_initialized()
    return jsonify(active_system.get_architecture_snapshot())


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
