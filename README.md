# Vocal Sheaf Analyzer Pro

A Unified Topological Framework for Neural Vocal Reconstruction. This tool analyzes AI-generated vocals using Sheaf Theory, Persistent Homology (TDA), and Non-Negative Matrix Factorization (NMF) to detect artifacts and ensure structural integrity.


1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   python -m streamlit run app.py
   ```

## Features

- **Topological Analysis**: Uses GUDHI for Persistent Homology to identify spectral holes and fragmentation.
- **Sheaf Gluing Index**: A proprietary metric (0-100%) measuring the consistency of the vocal manifold.
- **Signal Deconstruction**: Separates the 'Vocal Core' from 'Neural Residual' using NMF.
- **Benchmark Mode**: Compare the Sheaf Engine (RMVPE) against standard Harvest/Crepe baselines.
- **Presentation Mode**: Streamlined UI for quick pitches and jury demonstrations.

## Project Structure

- `app.py`: Main Streamlit Command Center.
- `analysis.py`: Core mathematical engine (TDA, Sheaf Graph, NMF).
- `model_loader.py`: RVC Inference wrapper with spectral enhancement logic.
- `data_simulation.py`: Neural artifact simulation (Phase Jitter, Bit-Crushing).
- `audio/`: Directory for input vocal profiles.
- `weights/`: RVC model weights and index files.

## Technical Metrics

- **MCD**: Mel-Cepstral Distortion (Timbral Match).
- **Phase Coherence**: Deviance in STFT phase alignment.
- **Latency**: End-to-end processing time in milliseconds.

---
*Developed for Advanced Neural Vocal Research & Industrial Auditing.*
