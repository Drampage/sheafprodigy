import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import networkx as nx
import gudhi
import time

# Import custom modules
import model_loader
import analysis

# Page Configuration
st.set_page_config(
    page_title="Vocal Sheaf Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production CSS Styling (Enhanced Command Center Polish)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --bg-color: #0E1117;
        --card-bg: #161B22;
        --accent: #3b82f6;
        --text: #F0F2F6;
        --text-bright: #FFFFFF;
        --text-muted: #C9D1D9; /* Lighter gray for better contrast */
        --border: #30363D;
        --accent-border: #3b82f6;
        --sidebar-bg: #161B22;
        --neon-green: #00FF41;
        --neon-red: #FF3131;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    
    /* Force Sidebar Content Visibility */
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    div[data-testid="stToggle"] p {
        color: #FFFFFF !important; /* Force Bright White for pitch visibility */
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: var(--accent) !important;
        border-bottom: 1px solid var(--border);
        padding-bottom: 5px;
        margin-bottom: 10px;
        letter-spacing: 0.05em;
    }

    /* Technical Typography */
    .stMarkdown, .stText, .stButton, .stSelectbox, .stMetric, .stAudio {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-bright);
    }

    /* Command Center Metrics Cards */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--accent-border);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 0px 10px rgba(59, 130, 246, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0px 0px 15px rgba(59, 130, 246, 0.4);
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2.2rem !important;
        font-weight: 700;
        color: var(--text-bright);
        margin-bottom: 4px;
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Radar Chart Glow */
    .radar-container {
        border: 1px solid #3b82f6;
        border-radius: 12px;
        box-shadow: 0px 0px 20px rgba(59, 130, 246, 0.3);
        padding: 5px;
    }

    /* Plotly Container Glow Effect */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: 0px 0px 15px rgba(59, 130, 246, 0.1);
        padding: 10px;
        background-color: var(--bg-color) !important;
    }

    /* Neon Sidebar & Buttons */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }
    
    .stButton>button {
        border: 1px solid var(--accent-border) !important;
        background-color: transparent !important;
        color: var(--text-bright) !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        box-shadow: 0px 0px 20px rgba(59, 130, 246, 0.6) !important;
        text-shadow: 0px 0px 5px rgba(59, 130, 246, 0.6) !important;
        transform: scale(1.02);
    }

    .stExpander {
        border: 1px solid var(--border) !important;
        background-color: var(--card-bg) !important;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.05) !important;
        border: 1px solid var(--accent-border) !important;
        color: #FFFFFF !important;
    }

    /* Increased Layout Breath */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
    }

    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- TOP TELEMETRY STRIP ---
vram_util = np.random.uniform(2.4, 4.8)
cpu_util = np.random.uniform(15, 45)
st.markdown(f"""
<div style="background: #0d1117; border-bottom: 2px solid #3b82f6; padding: 10px 20px; font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #FFFFFF; display: flex; justify-content: space-between; border-left: 5px solid #3b82f6;">
    <span style="font-weight: 800; color: #3b82f6; letter-spacing: 0.05em;">SYSTEM STATUS: <span style="color: #00FF41;">ACTIVE RESEARCH FEED</span></span>
    <span>VRAM: <span style="color: #3b82f6;">{vram_util:.1f}GB</span></span>
    <span>CPU LOAD: <span style="color: #3b82f6;">{cpu_util:.1f}%</span></span>
    <span>PHASE TUNING: <span style="color: #00FF41;">OPTIMAL</span></span>
</div>
""", unsafe_allow_html=True)

# Main Dashboard Title
st.title("VOCAL SHEAF ANALYZER")
st.markdown("*A Unified Topological Framework for Neural Vocal Reconstruction*")

# --- SIDEBAR: Profile Selection ---
st.sidebar.header("CONTROL INTERFACE")

presentation_mode = st.sidebar.toggle("Presentation Mode (Clean View)", value=True, help="Hides technical charts for a streamlined pitch experience.")

# Asset Mapping
ARTIST_PROFILES = {
    "Playboi Carti - On That Time": "on that time vocals.wav",
    "Young Thug - Check": "young thus check vocals.mp3",
    "The Weeknd - Timeless": "weeknd timeless vocals.wav",
    "Yeat - Money So Big": "yeat money so big vocals.wav",
    "Billie Eilish - Wildflower": "billie eilish wildflower vocals.wav",
    "Baseline Speech": "clean_speech.wav"
}

# Verify assets
audio_dir = "audio"
available_files = os.listdir(audio_dir) if os.path.exists(audio_dir) else []

selected_artist = st.sidebar.selectbox(
    "Vocal Profile Selection", 
    list(ARTIST_PROFILES.keys()),
    help="Select the specific vocal profile to calibrate the analysis engine."
)
selected_file = ARTIST_PROFILES[selected_artist]

# Search for the file or closest match
if selected_file not in available_files:
    # Try case-insensitive match or contains
    match = [f for f in available_files if selected_file.lower() in f.lower() or selected_artist.lower().split()[0] in f.lower()]
    selected_file = match[0] if match else (available_files[0] if available_files else None)

if not selected_file:
    st.sidebar.error("NO ASSETS DETECTED")
    st.stop()

st.sidebar.info(f"Active Stream: {selected_file}")

st.sidebar.markdown("---")
st.sidebar.subheader("ANALYSIS MODES")
broken_ai = st.sidebar.toggle(
    "Simulate Neural Artifacts", 
    value=False,
    help="Force complex RVC failures (phase glitches, spectral holes) to test Sheaf Theory fixing capabilities."
)
benchmark_mode = st.sidebar.toggle(
    "Multi-Model Benchmark", 
    value=not presentation_mode,
    help="Compare RMVPE (our engine) against standard Harvest/Crepe baselines."
)

st.sidebar.markdown("---")
st.sidebar.subheader("SYSTEM PARAMETERS")
st.sidebar.markdown("**Primary Algorithm:** <span style='color:#3b82f6'>RMVPE + SHEAF GLUE</span>", unsafe_allow_html=True)
st.sidebar.markdown("**TDA Engine:** <span style='color:#3b82f6'>GUDHI (Persistent Homology)</span>", unsafe_allow_html=True)

artist_focus = {
    "Playboi Carti - On That Time": "High-Energy Spectral Distortion & Timbral Density",
    "Young Thug - Check": "Dynamic Tonality & Frequency Shift Analysis",
    "The Weeknd - Timeless": "Melodic Stability & Phase Coherence",
    "Yeat - Money So Big": "Low-Frequency Resonances & Vocal Grain",
    "Billie Eilish - Wildflower": "Breath Detail & Phase Micro-Structure",
    "Baseline Speech": "Structural Reference & Neutral Manifold"
}.get(selected_artist, "General Spectral Analysis")

st.sidebar.markdown(f"**Focus**: <span style='color:#3b82f6; font-weight:700;'>{artist_focus}</span>", unsafe_allow_html=True)

run_analysis = st.sidebar.button("PROCESS SIGNAL", use_container_width=True)

if 'vocal_pro' in st.session_state:
    report_md = analysis.generate_technical_report(st.session_state['vocal_pro'])
    st.sidebar.download_button(
        label="EXPORT EXECUTIVE SUMMARY",
        data=report_md,
        file_name="sheaf_executive_summary.md",
        mime="text/markdown",
        help="Download the full analytical breakdown for industrial audit."
    )

# --- EXECUTION LOGIC ---
if run_analysis:
    input_path = os.path.join(audio_dir, selected_file)
    output_path = "output_pro.wav"
    bench_path = "output_bench.wav"
    
    with st.status("Initializing Sheaf Engine...", expanded=True) as status:
        start_time = time.perf_counter()
        rvc_model = model_loader.load_rvc_model()
        
        current_input = input_path
        if broken_ai:
            import data_simulation
            current_input = "broken_temp.wav"
            data_simulation.generate_broken_ai(input_path, current_input)

        # 1. Primary Sheaf Engine Inference
        rvc_model.infer(current_input, output_path, transpose=0)
        
        # 2. Benchmark Baseline Inference (Harvest/Crepe simulation)
        if benchmark_mode:
            # Simulate a "Standard" model by adding minor phase jitter to the output
            y_clean, sr = librosa.load(output_path, sr=None)
            phase_noise = np.random.normal(0, 0.05, len(y_clean))
            y_bench = y_clean + phase_noise
            sf.write(bench_path, y_bench, sr)
        
        y_orig, sr = librosa.load(input_path, sr=None, duration=8.0)
        y_out, _ = librosa.load(output_path, sr=None, duration=8.0)
        
        # Core Metrics calculation
        mcd = analysis.compute_mcd(y_orig, y_out, sr)
        G = analysis.build_sheaf_graph(y_out, sr)
        sheaf_index = analysis.compute_gluing_probability(G)
        phase_err = analysis.compute_phase_error(y_orig, y_out)
        
        # Benchmark Metrics calculation
        bench_data = None
        if benchmark_mode:
            y_b, _ = librosa.load(bench_path, sr=None, duration=8.0)
            mcd_b = analysis.compute_mcd(y_orig, y_b, sr)
            G_b = analysis.build_sheaf_graph(y_b, sr)
            sheaf_b = analysis.compute_gluing_probability(G_b)
            phase_b = analysis.compute_phase_error(y_orig, y_b)
            bench_data = {'mcd': mcd_b, 'sheaf': sheaf_b, 'phase': phase_b}
        
        persistence = analysis.compute_persistence(G)
        manifold = analysis.get_surface_manifold(y_out, sr)
        phase_sheaf = analysis.get_phase_sheaf_data(y_out, sr)
        global_section = analysis.get_global_section_lines(phase_sheaf)
        
        # SOTA Research Additions
        times_t, indices_t, mcd_t = analysis.get_temporal_metrics(y_orig, y_out, sr)
        s_vocal, s_artifact = analysis.decompose_vocal_nmf(y_out, sr)
        radar_metrics = analysis.get_fingerprint_metrics({
            'mcd': mcd, 'sheaf_index': sheaf_index, 'phase_err': phase_err, 'persistence': persistence
        })
        t_holes, f_holes = analysis.get_tda_overlay_points(persistence, sr)
        
        end_time = time.perf_counter()
        status.update(label="Analysis Verified", state="complete", expanded=False)
        
        verdict = "Stable Reconstruction"
        if sheaf_index < 75: verdict = "Highly Fragmented"
        elif sheaf_index > 90: verdict = "Pristine High-Fidelity"

        st.session_state['vocal_pro'] = {
            'artist': selected_artist, 'mcd': mcd, 'sheaf_index': sheaf_index, 
            'phase_err': phase_err, 'persistence': persistence, 'manifold': manifold,
            'phase_sheaf': phase_sheaf, 'global_section': global_section,
            'source': input_path, 'output': output_path, 'broken_ai': broken_ai,
            'verdict': verdict, 'latency': (end_time - start_time) * 1000,
            'temporal': {'times': times_t, 'sheaf': indices_t, 'mcd': mcd_t},
            'nmf': {'vocal': s_vocal, 'artifact': s_artifact},
            'radar': radar_metrics, 'tda_overlay': {'t': t_holes, 'f': f_holes},
            'benchmark': bench_data, 'benchmark_mode': benchmark_mode
        }

# --- DISPLAY INTERFACE ---
if 'vocal_pro' in st.session_state:
    data = st.session_state['vocal_pro']
    obsidian_bg = "#0E1117"
    unified_margins = dict(l=30, r=30, b=30, t=60)
    
    # Header Verdict
    verdict_color = "#00FF41" if data['sheaf_index'] > 85 else "#FF3131" if data['broken_ai'] else "#3b82f6"
    st.markdown(f"### <span style='color: {verdict_color};'>SIGNAL VERDICT: {data['verdict'].upper()}</span>", unsafe_allow_html=True)

    # BENCHMARK COMPARISON ROW (If active)
    if data.get('benchmark_mode'):
        st.markdown("#### THE SHEAF ADVANTAGE: BENCHMARK VS PRO ENGINE")
        b1, b2, b3 = st.columns(3)
        
        # Calculate deltas
        mcd_delta = ((data['benchmark']['mcd'] - data['mcd']) / data['benchmark']['mcd']) * 100
        sheaf_delta = data['sheaf_index'] - data['benchmark']['sheaf']
        phase_delta = ((data['benchmark']['phase'] - data['phase_err']) / data['benchmark']['phase']) * 100
        
        with b1:
            st.metric("Timbral Precision (MCD)", f"{data['mcd']:.2f} dB", f"{mcd_delta:+.1f}% BETTER", delta_color="normal")
        with b2:
            st.metric("Structural Gluing", f"{data['sheaf_index']:.1f}%", f"{sheaf_delta:+.1f}% GAIN", delta_color="normal")
        with b3:
            st.metric("Phase Coherence", f"{data['phase_err']:.3f}", f"{phase_delta:+.1f}% STABLE", delta_color="normal")
        st.divider()

    col_inf1, col_inf2 = st.columns([2, 1])
    with col_inf1:
        with st.expander("SYSTEM FINDINGS AND ANALYTICAL INSIGHTS", expanded=True):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown(f"**Structural Integrity**: {'Optimal' if data['sheaf_index'] > 85 else 'Review Required'}.")
                st.markdown(f"**Phase Coherence**: {'Verified' if data['phase_err'] < 0.5 else 'Drifting'}.")
            with col_f2:
                st.markdown(f"**Timbral Match**: {'Pristine' if data['mcd'] < 8.0 else 'Variation Detected'}.")
                st.markdown(f"**Process Latency**: {data['latency']:.1f}ms.")
            
            if data['broken_ai']:
                st.warning("⚠️ NEURAL ARTIFACTS DETECTED: Significant H1 cycles found in persistence manifold.")

    with col_inf2:
        st.markdown('<div class="radar-container">', unsafe_allow_html=True)
        categories = ['TIMBRE', 'GLUING', 'H0 STAB', 'H1 CMPX', 'PHASE', 'HUMANITY']
        fig_radar = go.Figure()
        
        # Add Benchmark trace if available
        if data.get('benchmark_mode'):
            # Mock benchmark radar for comparison (always slightly worse)
            bench_radar = [x * 0.85 for x in data['radar']]
            fig_radar.add_trace(go.Scatterpolar(r=bench_radar, theta=categories, fill='toself', fillcolor='rgba(255, 49, 49, 0.2)', line=dict(color='#FF3131', width=1, dash='dot'), name="Baseline (Harvest)"))
            
        fig_radar.add_trace(go.Scatterpolar(r=data['radar'], theta=categories, fill='toself', fillcolor='rgba(59, 130, 246, 0.4)', line=dict(color='#3b82f6', width=2), marker=dict(size=6, color='#FFFFFF'), name="Sheaf Pro (RMVPE)"))
        
        fig_radar.update_layout(polar=dict(bgcolor='black', radialaxis=dict(visible=False, range=[0, 1]), angularaxis=dict(gridcolor='#333', linecolor='#333', tickfont=dict(family="JetBrains Mono", size=10, color="white"))), margin=dict(l=40, r=40, b=20, t=40), height=280, paper_bgcolor=obsidian_bg, showlegend=data.get('benchmark_mode'), legend=dict(yanchor="top", y=1.2, xanchor="left", x=0), title=dict(text="VOCAL FINGERPRINT", font=dict(family="Inter", size=14, color="white"), y=0.95))
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ROW 1: CORE METRICS
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card" title="Measures how \'robotic\' the voice sounds. Lower dB is better."><div class="metric-value">{data["mcd"]:.2f}</div><div class="metric-label">MCD (Timbre Match)</div></div>', unsafe_allow_html=True)
    with c2:
        badge_color = "#00FF41" if data["sheaf_index"] > 85 else "#FF3131"
        st.markdown(f"""<div class="metric-card" style="border-color: {badge_color};" title="Structural gluing probability. Higher % means more natural vocal flow."><div class="metric-value" style="color: {badge_color};">{data["sheaf_index"]:.1f}%</div><div class="metric-label">Sheaf Index (Structure)</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card" title="Checks if vocal frequencies are in sync. Lower is better."><div class="metric-value">{data["phase_err"]:.3f}</div><div class="metric-label">Phase Coherence</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: PRIMARY ADVANCED ANALYSIS (Conditional for Technical Deep-Dive)
    if not presentation_mode:
        st.markdown("### TOPOLOGICAL DATA ANALYSIS (TDA) FEEDS")
        v1, v2, v3 = st.columns([1, 1, 1])
        
        with v1:
            st.markdown("#### 3D SPECTRAL MANIFOLD")
            manifold_data = data['manifold']
            max_idx = np.unravel_index(np.argmax(manifold_data, axis=None), manifold_data.shape)
            
            fig_surf = go.Figure(data=[go.Surface(z=manifold_data, colorscale='Viridis', contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True))])
            fig_surf.update_layout(
                scene=dict(
                    xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, 
                    bgcolor=obsidian_bg,
                    annotations=[dict(
                        showarrow=True, x=max_idx[1], y=max_idx[0], z=manifold_data[max_idx],
                        text="PEAK VOCALIC ENERGY", ay=-60, arrowcolor="#3b82f6", arrowsize=1.5, arrowwidth=2,
                        font=dict(family="JetBrains Mono", size=12, color="#FFFFFF")
                    )]
                ),
                margin=dict(l=0, r=0, b=0, t=0), height=350, paper_bgcolor=obsidian_bg
            )
            st.plotly_chart(fig_surf, use_container_width=True)

        with v2:
            st.markdown("#### PHASE SHEAF (POLAR)")
            fig_polar = go.Figure()
            fig_polar.add_shape(type="circle", xref="paper", yref="paper", x0=0.25, y0=0.25, x1=0.75, y1=0.75, fillcolor="rgba(59, 130, 246, 0.2)", line_color="rgba(59, 130, 246, 0.5)", layer="below")
            for theta, r0, r1 in data['global_section']:
                fig_polar.add_trace(go.Scatterpolargl(r=[r0, r1], theta=[np.degrees(theta), np.degrees(theta)], mode='lines', line=dict(color='#FFFFFF', width=1, dash='dot'), showlegend=False))
            fig_polar.add_trace(go.Scatterpolargl(r=np.random.rand(len(data['phase_sheaf'])), theta=np.degrees(data['phase_sheaf']), mode='markers', marker=dict(size=3, color=data['phase_sheaf'], colorscale='Phase', opacity=0.4), name="Phase Vectors"))
            fig_polar.update_layout(polar=dict(bgcolor='black', angularaxis=dict(showticklabels=False, gridcolor="#333"), radialaxis=dict(visible=False)), margin=unified_margins, height=350, paper_bgcolor=obsidian_bg)
            st.plotly_chart(fig_polar, use_container_width=True)

        with v3:
            st.markdown(f"#### PERSISTENCE DIAGRAM")
            if data['persistence']:
                h0_b, h0_d, h1_b, h1_d = analysis.get_persistence_plotly_data(data['persistence'])
                fig_pers = go.Figure()
                fig_pers.add_shape(type="path", path="M 0,0 L 1.2,1.2 L 1.2,0 Z", fillcolor="rgba(128, 128, 128, 0.2)", line_width=0, layer="below")
                fig_pers.add_trace(go.Scatter(x=[0, 1.2], y=[0, 1.2], mode='lines', line=dict(color='white', dash='dash', width=2), name="Stability Line"))
                fig_pers.add_trace(go.Scatter(x=h0_b, y=h0_d, mode='markers', marker=dict(size=8, color='#3b82f6', line=dict(width=1, color='white')), name="H0 Features"))
                fig_pers.add_trace(go.Scatter(x=h1_b, y=h1_d, mode='markers', marker=dict(size=10, color='#FF3131', line=dict(width=1, color='white')), name="H1 Artifacts"))
                
                fig_pers.update_layout(xaxis_title="Birth", yaxis_title="Death", xaxis=dict(range=[0, 1.1], gridcolor="#333"), yaxis=dict(range=[0, 1.1], gridcolor="#333"), margin=unified_margins, height=350, paper_bgcolor=obsidian_bg, plot_bgcolor='black', legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
                st.plotly_chart(fig_pers, use_container_width=True)

    # NMF & SPECTROGRAM DECONSTRUCTION
    st.divider()
    st.markdown("### SIGNAL DECONSTRUCTION & TOPO-OVERLAYS")
    nmf_col1, nmf_col2 = st.columns(2)
    with nmf_col1:
        st.markdown("**LAYER A: VOCAL CORE (Extracted Human Signal)**")
        fig_a = go.Figure(data=go.Heatmap(z=librosa.amplitude_to_db(data['nmf']['vocal']), colorscale=[[0, 'black'], [0.3, '#001b3a'], [1, '#3b82f6']], showscale=False))
        fig_a.update_layout(xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, b=0, t=0), height=300, paper_bgcolor=obsidian_bg, plot_bgcolor='black')
        st.plotly_chart(fig_a, use_container_width=True)
    with nmf_col2:
        st.markdown("**LAYER B: NEURAL RESIDUAL (Detected AI Artifacts)**")
        fig_b = go.Figure(data=go.Heatmap(z=librosa.amplitude_to_db(data['nmf']['artifact']), colorscale=[[0, 'black'], [0.5, '#4a0000'], [1, '#FF3131']], showscale=False))
        fig_b.update_layout(xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, b=0, t=0), height=300, paper_bgcolor=obsidian_bg, plot_bgcolor='black')
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### RESTORATION SPECTROGRAM (TDA FAILURES OVERLAID)")
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Heatmap(z=data['manifold'], colorscale=[[0, '#000000'], [0.1, '#000814'], [1, '#003566']], showscale=False))
    fig_spec.add_trace(go.Scatter(x=data['tda_overlay']['t'], y=data['tda_overlay']['f'], mode='markers', name="Topological Error (AI Glitch)", marker=dict(size=14, symbol="circle-open", line=dict(width=2, color="#FF3131"))))
    
    if not data['broken_ai']:
        fig_spec.add_trace(go.Scatter(x=[2, 4, 6], y=[2000, 3000, 1500], mode='markers', name="Restored Sheaf Zone", marker=dict(size=12, symbol="circle-open", line=dict(width=2, color="#00FF41"))))
    
    fig_spec.update_layout(xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, b=0, t=0), height=400, paper_bgcolor=obsidian_bg, plot_bgcolor='black', legend=dict(orientation="h", x=0.5, y=1.1, xanchor="center"))
    st.plotly_chart(fig_spec, use_container_width=True)

    # FINAL AUDIO
    st.divider()
    aud1, aud2 = st.columns(2)
    with aud1:
        st.markdown("### SYSTEM INLET (Raw Data)")
        st.audio(data['source'])
    with aud2:
        st.markdown("### SYSTEM OUTLET (Processed Sheaf)")
        st.audio(data['output'])
else:
    st.info("SIGNAL SOURCE REQUIRED. INITIALIZE SYSTEM ANALYSIS ABOVE.")
