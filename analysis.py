"""
Sheaf Theory Analysis Module
Implements topological analysis of audio signals using Sheaf Theory and TDA.
"""

import numpy as np
import librosa
import networkx as nx
from sklearn.decomposition import NMF
import gudhi
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

def decompose_vocal_nmf(y, sr, n_components=2):
    """
    Decomposes the signal into 'Vocal Core' and 'Neural Residual' using NMF.
    
    Args:
        y: Audio signal
        sr: Sample rate
        
    Returns:
        S_vocal, S_artifact: Separated spectrograms
    """
    S = np.abs(librosa.stft(y))
    model = NMF(n_components=n_components, init='nndsvd', random_state=42)
    W = model.fit_transform(S)
    H = model.components_
    
    # Simple heuristic: component with more low-frequency energy is usually vocal
    vocal_idx = 0
    if np.sum(W[:W.shape[0]//4, 1]) > np.sum(W[:W.shape[0]//4, 0]):
        vocal_idx = 1
    artifact_idx = 1 - vocal_idx
    
    # Reconstruct separated spectrograms
    S_vocal = np.outer(W[:, vocal_idx], H[vocal_idx, :])
    S_artifact = np.outer(W[:, artifact_idx], H[artifact_idx, :])
    
    return S_vocal, S_artifact


def build_sheaf_graph(y, sr, frame_ms=50):
    """
    Constructs a Sheaf graph representing topological structure of audio.
    
    Nodes: Audio fragments (~50ms chunks)
    Edges: Spectral consistency (Gluing Property)
    
    Uses MFCC + Spectral Centroid for robust feature representation.
    
    Args:
        y: Audio signal
        sr: Sample rate
        frame_ms: Frame size in milliseconds
        
    Returns:
        NetworkX graph representing the audio sheaf
    """
    
    # Extract multiple features for robust topology
    # 1. MFCCs (timbral features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    
    # 2. Spectral Centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    
    # 3. Spectral Rolloff (frequency distribution)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
    
    # Combine features
    features = np.vstack([mfcc, centroid, rolloff]).T
    times = librosa.times_like(mfcc, sr=sr, hop_length=512)
    
    # Ensure alignment
    n_frames = min(len(features), len(times))
    features = features[:n_frames]
    times = times[:n_frames]
    
    # Create graph
    G = nx.Graph()
    
    # Safety check
    if n_frames < 2:
        G.add_node(0, time=0.0, feature=np.zeros(features.shape[1]))
        return G
    
    # Downsample to ~20ms nodes for denser TDA cloud
    hop = max(1, int(0.02 * sr / 512))  # Convert 20ms to frame count
    
    node_indices = []
    for i in range(0, n_frames, hop):
        G.add_node(i, time=times[i], feature=features[i])
        node_indices.append(i)
    
    # Edge construction: Implement Sheaf Gluing Property
    # Connect nodes based on spectral similarity (topological consistency)
    
    def compute_similarity(feat1, feat2):
        """Compute multi-metric similarity"""
        # Cosine similarity
        cos_sim = 1.0 - cosine(feat1, feat2)
        
        # Euclidean distance (normalized)
        euc_dist = euclidean(feat1, feat2)
        euc_sim = 1.0 / (1.0 + euc_dist)
        
        # Combined metric
        return (cos_sim + euc_sim) / 2.0
    
    # Dynamic threshold based on feature statistics
    all_similarities = []
    for i in range(min(100, len(node_indices) - 1)):
        u_idx = node_indices[i]
        v_idx = node_indices[i + 1]
        sim = compute_similarity(features[u_idx], features[v_idx])
        all_similarities.append(sim)
    
    if len(all_similarities) > 0:
        mean_sim = np.mean(all_similarities)
        std_sim = np.std(all_similarities)
        threshold_high = mean_sim + 0.5 * std_sim
        threshold_low = mean_sim - 0.5 * std_sim
    else:
        threshold_high = 0.85
        threshold_low = 0.65
    
    # Build edges
    edges_added = 0
    
    for i_idx in range(len(node_indices) - 1):
        u_idx = node_indices[i_idx]
        v_idx = node_indices[i_idx + 1]
        
        # 1. Temporal continuity (always connect neighbors)
        sim = compute_similarity(features[u_idx], features[v_idx])
        G.add_edge(u_idx, v_idx, weight=sim, edge_type='temporal')
        edges_added += 1
        
        # 2. Topological gluing (non-local connections)
        # Look ahead in a window
        window_size = min(8, len(node_indices) - i_idx)
        
        for offset in range(2, window_size):
            target_idx = node_indices[i_idx + offset]
            
            # Compute similarity
            sim = compute_similarity(features[u_idx], features[target_idx])
            
            # Connect if highly similar (Sheaf gluing condition)
            if sim > threshold_high:
                G.add_edge(u_idx, target_idx, weight=sim, edge_type='gluing')
                edges_added += 1
    
    # If graph is too sparse, lower threshold
    if edges_added < len(node_indices) * 1.5:
        for i_idx in range(len(node_indices) - 1):
            u_idx = node_indices[i_idx]
            
            window_size = min(6, len(node_indices) - i_idx)
            for offset in range(2, window_size):
                target_idx = node_indices[i_idx + offset]
                
                if not G.has_edge(u_idx, target_idx):
                    sim = compute_similarity(features[u_idx], features[target_idx])
                    
                    if sim > threshold_low:
                        G.add_edge(u_idx, target_idx, weight=sim, edge_type='gluing_soft')
    
    return G


def compute_persistence(G):
    """
    Computes Persistent Homology using GUDHI.
    
    This reveals the topological structure (Global Section) of the audio sheaf.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Persistence intervals
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return []
        
    st = gudhi.SimplexTree()
    
    # Insert vertices
    for node in G.nodes():
        st.insert([node], filtration=0.0)
    
    # Insert edges with filtration based on dissimilarity
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0.0)
        # Lower filtration = appears earlier = stronger connection
        filtration = max(0.0, 1.0 - weight)
        st.insert([u, v], filtration=filtration)
    
    # Compute persistence
    st.compute_persistence()
    persistence = st.persistence()
    
    return persistence


def get_persistence_plotly_data(persistence):
    """
    Formats persistence data specifically for Plotly visualization.
    """
    h0_birth = []
    h0_death = []
    h1_birth = []
    h1_death = []
    
    for p in persistence:
        dim = p[0]
        birth, death = p[1]
        # Replace infinity with a large value for plotting
        if death == float('inf'):
            death = 1.2 
            
        if dim == 0:
            h0_birth.append(birth)
            h0_death.append(death)
        elif dim == 1:
            h1_birth.append(birth)
            h1_death.append(death)
            
    return h0_birth, h0_death, h1_birth, h1_death



def compute_mcd(y1, y2, sr):
    """
    Compute Mel-Cepstral Distortion between two signals.
    """
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)
    
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]
    
    diff = mfcc1 - mfcc2
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
    
    return mcd

def get_temporal_metrics(y_orig, y_out, sr, window_sec=0.5):
    """
    Pinpoints signal failures in time by calculating metrics over sliding windows.
    
    Returns:
        times, sheaf_indices, mcd_values
    """
    win_len = int(window_sec * sr)
    hop_len = win_len // 2
    
    sheaf_indices = []
    mcd_values = []
    times = []
    
    for start in range(0, len(y_out) - win_len, hop_len):
        end = start + win_len
        window_out = y_out[start:end]
        window_orig = y_orig[start:end]
        
        # MCD for this window
        mcd_val = compute_mcd(window_orig, window_out, sr)
        mcd_values.append(mcd_val)
        
        # Sheaf Index for this window
        G_win = build_sheaf_graph(window_out, sr)
        sheaf_val = compute_gluing_probability(G_win)
        sheaf_indices.append(sheaf_val)
        
        times.append(start / sr)
        
    return np.array(times), np.array(sheaf_indices), np.array(mcd_values)


def compute_gluing_probability(G):
    """
    Calculates the average probability of successful gluing across the sheaf.
    Based on edge weights (similarities).
    """
    if G.number_of_edges() == 0:
        return 0.0
    
    weights = [data.get('weight', 0) for u, v, data in G.edges(data=True)]
    return np.mean(weights) * 100

def compute_phase_error(y1, y2):
    """
    Calculates structural integrity via Phase Error (STFT Phase Deviation).
    Lower is better.
    """
    D1 = librosa.stft(y1)
    D2 = librosa.stft(y2)
    
    # Align
    min_len = min(D1.shape[1], D2.shape[1])
    p1 = np.angle(D1[:, :min_len])
    p2 = np.angle(D2[:, :min_len])
    
    # Mean absolute circular difference
    phase_diff = np.abs(np.mod(p1 - p2 + np.pi, 2 * np.pi) - np.pi)
    return np.mean(phase_diff)

def get_surface_manifold(y, sr):
    """
    Generates a 3D Surface Plot data (Frequency-Time-Energy).
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Downsample for performance
    S_db = S_db[::4, ::4]
    
    return S_db

def get_phase_sheaf_data(y, sr):
    """
    Extracts phase angles for polar plot visualization (Phase Sheaf).
    """
    D = librosa.stft(y)
    angles = np.angle(D)
    
    # Downsample for visualization
    phase_sample = angles[::8, ::8].flatten()
    return phase_sample

def get_global_section_lines(phase_data, n_lines=12):
    """
    Computes prominent phase angles representing 'Global Section' alignment.
    """
    # Use a histogram to find prominent phase regions
    hist, bin_edges = np.histogram(phase_data, bins=36, range=(-np.pi, np.pi))
    
    # Find peaks in the histogram
    peaks = bin_edges[np.argsort(hist)[-n_lines:]]
    
    # Return (theta, r_start, r_end) for drawing lines
    return [(p, 0, 1.0) for p in peaks]


def generate_technical_report(data):
    """
    Generates a Markdown report brief for industrial auditing.
    """
    import time
    report = f"""# INDUSTRIAL ANALYSIS REPORT: VOCAL SHEAF INTEGRITY
## PROFILE: {data['artist'].upper()}
## TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}

### 1. ACOUSTIC METRICS
- **Mel-Cepstral Distortion (MCD)**: {data['mcd']:.2f} dB
- **Phase Structural Error**: {data['phase_err']:.4f}
- **Processing Latency**: {data['latency']:.1f} ms

### 2. TOPOLOGICAL DIAGNOSTICS
- **Sheaf Index (Gluing Probability)**: {data['sheaf_index']:.1f}%
- **Artifact Detection**: {"POSITIVE" if data['broken_ai'] else "NEGATIVE"}
- **Structural Verdict**: {data.get('verdict', 'VERIFIED')}

### 3. SYSTEM TELEMETRY (SIMULATED)
- **VRAM Utilization**: {np.random.uniform(2.4, 4.8):.1f} GB
- **CPU Load**: {np.random.uniform(15, 45):.1f}%
- **Bandwidth**: 48.0 kHz 

---
*Generated by Vocal Sheaf Analyzer Pro - Proprietary Tool Chain*
"""
    return report

def get_fingerprint_metrics(data):
    """
    Aggregates and normalizes 6 key dimensions for the Radar Chart.
    """
    # Dimensions: MCD(Inv), SheafIndex, H0_Stability, H1_Complexity, Phase_Coherence, Spectral_Flux
    
    # Normalize MCD (0-15 dB range typical, we want higher is better)
    mcd_norm = max(0, 1.0 - (data['mcd'] / 15.0))
    
    # Sheaf Index (0-100)
    sheaf_norm = data['sheaf_index'] / 100.0
    
    # TDA Stability (H0 count vs total points)
    persistence = data['persistence']
    h0_count = len([p for p in persistence if p[0] == 0])
    h1_count = len([p for p in persistence if p[0] == 1])
    total = max(1, h0_count + h1_count)
    h0_norm = h0_count / total
    h1_norm = min(1.0, h1_count / 10.0) # H1 is usually small
    
    # Phase Coherence (Lower error is better)
    phase_norm = max(0, 1.0 - data['phase_err'])
    
    # Synthetic "Humanity" score
    humanity = (mcd_norm + sheaf_norm + h0_norm) / 3.0
    
    return [mcd_norm, sheaf_norm, h0_norm, h1_norm, phase_norm, humanity]

def get_tda_overlay_points(persistence, sr):
    """
    Identifies 'topological holes' (H1 points) to overlay on spectrogram.
    Returns (times, frequencies) of significant holes.
    """
    holes = [p[1] for p in persistence if p[0] == 1]
    
    # Map persistence values to something visual
    # This is a heuristic mapping for demonstration
    t_points = []
    f_points = []
    
    for birth, death in holes:
        if (death - birth) > 0.05: # Only significant holes
            t_points.append(birth * 5.0) # Scaled to audio length
            f_points.append(1000 + np.random.uniform(-200, 200)) # Targeted at mid-range
            
    return t_points, f_points
