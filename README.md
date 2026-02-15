# Vers3Dynamics Search
### Real-Time RF Spectrum Situational Awareness System

A sophisticated electromagnetic spectrum monitoring platform that provides real-time 3D visualization of radio-frequency energy, pattern detection, anomaly identification, and temporal stability analysis.

![Vers3Dynamics Search](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Vers3Dynamics Search transforms invisible electromagnetic activity into an intuitive 3D situational awareness display. The system continuously monitors the RF spectrum, detects anomalies, and visualizes the dynamic operational environment in real-time.

### Key Features

- **Real-time Spectrum Monitoring**: Continuous ingestion and processing of RF power data across 64 frequency bands (88 MHz - 2.4 GHz)
- **Anomaly Detection**: Statistical baseline modeling with automatic detection of unusual spectral activity
- **3D Visualization**: Interactive browser-based interface with spatial field representation
- **Temporal Analysis**: Historical pattern tracking and stability metrics
- **Situational Awareness**: Observer-centric perspective showing the surrounding EM environment
- **Live Telemetry Panel**: Real-time rollups for average power, strongest active band, anomaly ratio, and update timestamp
- **System Health API**: Extended `/api/status` telemetry including uptime and configured sample rate

---

## Technical Architecture

### Core Components

1. **RF Processor**: Signal processing pipeline with FFT analysis, band aggregation, and feature extraction
2. **Anomaly Detection**: Statistical modeling using z-scores and variance thresholding
3. **Spectral Nodes**: Dynamic representation of frequency bands with temporal memory
4. **3D Visualization Engine**: Plotly-based rendering with real-time updates
5. **Web Server**: Flask backend serving the interactive interface

### Signal Processing Pipeline

```
SDR/Simulator → Band Aggregation → Baseline Computation → Anomaly Detection → 3D Rendering
```

### Data Flow

```
RF Energy (continuous)
    ↓
Frequency Bands (64 channels)
    ↓
Temporal Windows (100 samples)
    ↓
Statistical Features (power, variance, stability)
    ↓
3D Spatial Field (interactive visualization)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Edge)

### Setup

1. **Clone or download the files**
   ```bash
   # Place vers3dynamics_search.py and requirements.txt in the same directory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python vers3dynamics_search.py
   ```

4. **Access the interface**
   - Open your browser to `http://localhost:5000`
   - The 3D visualization will initialize automatically

---

## Configuration

### System Parameters

Edit the `Config` class in `vers3dynamics_search.py`:

```python
class Config:
    # Frequency spectrum
    FREQ_START = 88e6      # Start frequency (Hz)
    FREQ_END = 2.4e9       # End frequency (Hz)
    NUM_BANDS = 64         # Number of monitoring bands
    
    # Temporal parameters
    SAMPLE_RATE = 10       # Samples per second
    WINDOW_SIZE = 100      # History buffer size
    UPDATE_INTERVAL = 0.1  # Visualization update rate
    
    # Anomaly detection
    ANOMALY_THRESHOLD = 2.5    # Sigma threshold
    STABILITY_THRESHOLD = 0.15  # Variance threshold
    
    # Visualization
    PORT = 5000            # Web server port
```

### SDR Integration

To connect a real SDR device:

1. Set `SIMULATE_SDR = False` in the Config class
2. Implement the SDR interface in the `SpectrumSource` class:

```python
def get_spectrum_snapshot(self) -> np.ndarray:
    # Replace simulation with actual SDR reading
    # Example for RTL-SDR:
    # samples = sdr.read_samples(256*1024)
    # spectrum = np.fft.fftshift(np.abs(np.fft.fft(samples)))
    # return 10 * np.log10(spectrum)
    pass
```

---

## Visualization Guide

### 3D Display Elements

- **Spectral Nodes**: Spheres representing frequency bands
  - **Size**: Proportional to signal power
  - **Color**: 
    - **Cyan**: Low frequency (VHF/UHF)
    - **Lime**: Mid frequency (L/S Band)
    - **Magenta**: High frequency (C/X Band)
    - **Red**: Anomaly detected (>2.5σ deviation)
  - **Opacity**: Indicates temporal stability (solid = stable, transparent = volatile)

- **Observer Platform**: Yellow diamond at origin (your sensor location)

- **Spatial Arrangement**: Cylindrical coordinate system
  - **Angle**: Frequency distribution around observer
  - **Radius**: Frequency domain separation
  - **Height**: Vertical frequency mapping (low→high)

### Interaction

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Hover**: View detailed band information

---

## Understanding the Data

### Metrics Explained

1. **Power (dBm)**: Current signal strength in the frequency band
2. **Baseline**: Historical median power level
3. **Anomaly Score**: Standard deviations from baseline (z-score)
4. **Stability**: Temporal consistency (0% = highly variable, 100% = stable)
5. **Band Class**: Frequency range classification

### Anomaly Detection

The system flags anomalies when:
- Power exceeds baseline by >2.5 standard deviations
- Sudden energy spikes appear
- Unusual temporal patterns emerge

### Temporal Stability

Stability metrics help identify:
- **High stability** (>80%): Consistent background signals
- **Medium stability** (40-80%): Intermittent communications
- **Low stability** (<40%): Transient or noisy activity

---

## Use Cases

### 1. Spectrum Monitoring
Monitor RF environment for authorized and unauthorized transmissions

### 2. Interference Detection
Identify unexpected signals that may interfere with operations

### 3. Pattern Analysis
Track temporal patterns in spectrum usage

### 4. Situational Awareness
Maintain real-time understanding of the EM operational environment

### 5. Research & Development
Study RF propagation, signal characteristics, and spectral behavior

---

## Architecture Details

### Spectral Node Structure

```python
@dataclass
class SpectralNode:
    frequency: float        # Center frequency (Hz)
    power: float           # Current power (dBm)
    baseline: float        # Historical baseline
    variance: float        # Temporal variance
    anomaly_score: float   # Deviation metric
    stability: float       # Stability index (0-1)
    band_class: str        # Frequency classification
    position: Tuple        # 3D coordinates
    history: deque         # Rolling buffer
```

### Processing Workflow

1. **Acquisition**: Sample RF spectrum at configured rate
2. **Aggregation**: Organize power data into frequency bands
3. **Analysis**: Compute statistical features and baselines
4. **Detection**: Identify anomalies using threshold criteria
5. **Visualization**: Render 3D scene with current state
6. **Update**: Push changes to web interface

---

## Performance Considerations

- **Update Rate**: 10 Hz acquisition, 2 Hz visualization refresh
- **Memory**: ~50 MB for 100-sample history across 64 bands
- **CPU**: <5% on modern processors
- **Network**: Minimal bandwidth (<100 KB/s for web updates)

---

## Troubleshooting

### Port Already in Use
```bash
# Change port in Config class
PORT = 5001  # Use different port
```

### Dependencies Not Installing
```bash
# Try upgrading pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Visualization Not Updating
- Check browser console for errors
- Verify Flask server is running
- Check firewall settings for localhost:5000

### Performance Issues
- Reduce `NUM_BANDS` (e.g., 32 instead of 64)
- Increase `UPDATE_INTERVAL` (e.g., 0.2 instead of 0.1)
- Decrease `SAMPLE_RATE` (e.g., 5 instead of 10)

---

## Development

### Adding Custom Signal Sources

Modify the `_initialize_sources()` method:

```python
def _initialize_sources(self):
    sources = [
        SignalSource(
            frequency=915e6,      # 915 MHz ISM band
            bandwidth=2e6,        # 2 MHz wide
            power=-50,            # -50 dBm
            modulation='FSK',
            duty_cycle=0.8,
            phase=0
        ),
        # Add more sources...
    ]
    return sources
```

### Customizing Visualization

Edit the `create_3d_scene()` method in the `Visualizer` class to:
- Change color schemes
- Modify node shapes
- Adjust spatial arrangement
- Add additional traces or annotations

---

## Future Enhancements

- [ ] Support for additional SDR hardware (HackRF, Airspy, etc.)
- [ ] Machine learning-based anomaly detection
- [ ] Recording and playback of spectrum sessions
- [ ] Multi-observer distributed monitoring
- [ ] Export to standard spectrum analysis formats
- [ ] Advanced signal classification
- [ ] Historical trend analysis and reporting

---

## License

MIT License - See code header for full license text

---

## Technical Support

For issues, questions, or contributions, please refer to the code comments and inline documentation. The system is designed to be self-explanatory for developers familiar with RF systems and Python.

---

## Acknowledgments

Built with:
- **NumPy**: Numerical computing
- **Plotly**: 3D visualization
- **Flask**: Web framework
- **Python**: Core language

---

## Disclaimer

This software is intended for authorized spectrum monitoring and research purposes only. Users are responsible for compliance with local regulations regarding RF monitoring and spectrum access. The simulated data mode is provided for demonstration and development purposes.

---

**Vers3Dynamics Search** - *Illuminating the Invisible Spectrum*


## Deployment (Vercel)

This project now includes a dedicated Flask entrypoint file (`app.py`) so Vercel can auto-detect the application.

- Vercel will discover `app.py` and the exported Flask object `app`
- The RF system initialization is handled lazily/once via `ensure_system_initialized()` for API routes
- No extra framework-specific bootstrap file is required
