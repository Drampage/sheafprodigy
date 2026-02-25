import librosa
import soundfile as sf
import numpy as np
import random
import scipy.signal
from scipy.signal import butter, sosfilt

def generate_broken_ai(input_path, output_path):
    """
    Simulates 'Broken AI' (RVC Failure) with ACTUAL neural network artifacts.
    NO chipmunk effect - instead we simulate:
    1. Phase Randomization (Metallic/Robotic jitter)
    2. Bit-Crushing (Quantization noise from failing AI decoder)
    3. Spectral Dropping (Frequency band dropouts)
    4. Granular Synthesis artifacts (Falling apart effect)
    """
    y, sr = librosa.load(input_path, sr=None)
    
    # STEP 1: Granular Synthesis Artifacts (Time-domain fragmentation)
    # Simulate neural network "stuttering" by creating micro-repeats and dropouts
    grain_size = int(sr * 0.02)  # 20ms grains
    n_grains = len(y) // grain_size
    y_granular = np.zeros_like(y)
    
    for i in range(n_grains):
        start = i * grain_size
        end = min(start + grain_size, len(y))
        grain = y[start:end]
        
        # Randomly apply artifacts to grains
        artifact = np.random.random()
        if artifact < 0.15:
            # Dropout (silence)
            grain = grain * 0.0
        elif artifact < 0.30:
            # Repeat previous grain (stuttering)
            if i > 0:
                prev_start = (i-1) * grain_size
                prev_end = min(prev_start + grain_size, len(y))
                grain = y[prev_start:prev_end]
        elif artifact < 0.45:
            # Reverse grain (glitch)
            grain = grain[::-1]
        
        y_granular[start:end] = grain[:len(y_granular[start:end])]
    
    y = y_granular
    
    # STEP 2: Phase Randomization (STFT domain corruption)
    # This creates the "metallic/robotic" sound typical of RVC failures
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)
    
    # Add random phase noise across ALL frequencies (not just high)
    # This simulates phase reconstruction errors in neural vocoders
    phase_corruption = np.random.uniform(-np.pi, np.pi, phase.shape)
    phase_corruption = np.exp(1j * phase_corruption)
    
    # Mix original phase with random phase (70% corruption for strong effect)
    corrupted_phase = phase * 0.3 + phase_corruption * 0.7
    corrupted_phase = corrupted_phase / (np.abs(corrupted_phase) + 1e-10)
    
    # STEP 3: Spectral Dropping (Frequency band dropouts)
    # Randomly "kill" frequency bins to simulate spectral reconstruction failures
    n_dropouts = 40
    for _ in range(n_dropouts):
        if magnitude.shape[1] <= 5:
            break
        
        # Random frequency band
        f_start = np.random.randint(0, magnitude.shape[0] - 20)
        f_width = np.random.randint(5, 20)
        
        # Random time segment
        t_start = np.random.randint(0, max(1, magnitude.shape[1] - 10))
        t_width = np.random.randint(3, 10)
        
        # Kill this spectral region
        magnitude[f_start:f_start+f_width, t_start:min(t_start+t_width, magnitude.shape[1])] *= 0.05
    
    # Add spectral "bursts" (over-reconstruction artifacts)
    n_bursts = 20
    for _ in range(n_bursts):
        if magnitude.shape[1] <= 5:
            break
            
        f_start = np.random.randint(0, magnitude.shape[0] - 15)
        f_width = np.random.randint(5, 15)
        t_start = np.random.randint(0, max(1, magnitude.shape[1] - 8))
        t_width = np.random.randint(2, 8)
        
        # Amplify this region (neural network hallucination)
        magnitude[f_start:f_start+f_width, t_start:min(t_start+t_width, magnitude.shape[1])] *= 4.0
    
    # Reconstruct from corrupted STFT
    y_phase_corrupted = librosa.istft(magnitude * corrupted_phase, hop_length=hop_length)
    
    # STEP 4: Bit-Crushing (Quantization noise)
    # Simulate low-quality AI decoder output
    bit_depth = 6  # Reduce to 6-bit (64 levels)
    max_val = np.max(np.abs(y_phase_corrupted))
    if max_val > 0:
        y_normalized = y_phase_corrupted / max_val
        # Quantize
        n_levels = 2 ** bit_depth
        y_quantized = np.round(y_normalized * (n_levels / 2)) / (n_levels / 2)
        # Scale back
        y_bitcrushed = y_quantized * max_val
    else:
        y_bitcrushed = y_phase_corrupted
    
    # STEP 5: Add subtle high-frequency noise (neural network hiss)
    noise = np.random.randn(len(y_bitcrushed)) * 0.01
    # High-pass the noise
    sos = butter(4, 4000, 'hp', fs=sr, output='sos')
    noise_filtered = sosfilt(sos, noise)
    
    y_broken = y_bitcrushed + noise_filtered
    
    # Normalize to prevent clipping
    y_broken = librosa.util.normalize(y_broken)
    
    sf.write(output_path, y_broken, sr)
    print(f"Generated broken AI output: {output_path}")
    return output_path

def generate_sheaf_fixed(input_path, output_path):
    """
    Simulates 'Sheaf Reconstruction'.
    Does NOT actually use Sheaf theory for generation (that's for analysis),
    but creates a 'fixed' sounding version using DSP.
    1. High-pass filter to remove mud.
    2. Normalize.
    """
    y, sr = librosa.load(input_path, sr=None)
    
    # 1. High-pass filter (soft clean)
    sos = butter(10, 100, 'hp', fs=sr, output='sos')
    y_filtered = sosfilt(sos, y)
    
    # 2. Normalize
    y_fixed = librosa.util.normalize(y_filtered)
    
    sf.write(output_path, y_fixed, sr)
    print(f"Generated sheaf-fixed output: {output_path}")
    return output_path
