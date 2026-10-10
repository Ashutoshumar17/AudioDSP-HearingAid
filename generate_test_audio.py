"""
generate_test_audio.py
Creates a synthetic hearing-aid test scenario: simulated speech-like signal
corrupted by common real-world noise sources.

Signal:  Formant-based vowel sounds (simulates human speech fundamentals)
Noise:   50/60 Hz electrical hum + crowd babble noise + broadband hiss
"""

import numpy as np
from src.io_utils import save_audio

fs = 16000          # sample rate (standard for hearing-aid testing)
duration = 5        # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# ---- Clean speech-like signal ----
# Simulate speech using fundamental frequency modulation + formants
# Fundamental frequency varies between 100-200 Hz (typical human voice)
     
f0 = 150 + 30 * np.sin(2 * np.pi * 2 * t)          # pitch modulation
phase = 2 * np.pi * np.cumsum(f0) / fs
glottal_pulse = 0.3 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)

# Add formant-like resonances (vowel /a/: F1~700Hz, F2~1200Hz, F3~2500Hz)
formant1 = 0.20 * np.sin(2 * np.pi * 700 * t)
formant2 = 0.15 * np.sin(2 * np.pi * 1200 * t)
formant3 = 0.10 * np.sin(2 * np.pi * 2500 * t)

# Amplitude envelope to mimic syllable-like bursts
envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))   # ~3 syllables/sec
clean_signal = envelope * (glottal_pulse + formant1 + formant2 + formant3)

# Normalize clean signal
clean_signal = clean_signal / np.max(np.abs(clean_signal)) * 0.5

# ---- Noise sources ----
# 1. Electrical hum (50 Hz + harmonics, common in hearing aid environments)
hum = (0.10 * np.sin(2 * np.pi * 50 * t)
     + 0.06 * np.sin(2 * np.pi * 100 * t)
     + 0.03 * np.sin(2 * np.pi * 150 * t))

# 2. Crowd/babble noise (band-limited random noise 200-4000 Hz)
np.random.seed(42)
babble = 0.08 * np.random.normal(size=len(t))

# simple low-pass to band-limit the babble
from scipy.signal import butter, filtfilt
b, a = butter(4, [200, 4000], btype='band', fs=fs)
babble = filtfilt(b, a, babble)

# 3. Broadband hiss (high-freq environmental noise)
hiss = 0.04 * np.random.normal(size=len(t))

# ---- Combine ----
noisy_signal = clean_signal + hum + babble + hiss

save_audio("data/raw/noisy_sample.wav", noisy_signal, fs)
save_audio("data/raw/clean_reference.wav", clean_signal, fs)

print("Created hearing-aid test files:")
print("  data/raw/noisy_sample.wav    (speech + hum + babble + hiss)")
print("  data/raw/clean_reference.wav (clean speech reference)")
