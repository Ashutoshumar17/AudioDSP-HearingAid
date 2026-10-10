"""
equalizer.py
5-band graphic equalizer built from band-pass filters.
Each band is filtered separately, scaled by a gain, then summed back together.

Includes clinical hearing-loss presets (audiogram-based gain profiles).
"""

import numpy as np
from src.filters import butter_bandpass, apply_filter

# (low_hz, high_hz) for each band
BANDS = [
    (20, 250),      # bass
    (250, 500),     # low-mid
    (500, 2000),    # mid
    (2000, 6000),   # high-mid
    (6000, 20000),  # treble
]

BAND_NAMES = ["Bass", "Low-Mid", "Mid", "High-Mid", "Treble"]

# ---- Clinical Hearing-Loss Presets ----
# Gains derived from common audiogram patterns:
# Each preset boosts frequencies where hearing loss is typical.
HEARING_PROFILES = {
    "Normal Hearing": {
        "description": "No amplification — flat response for normal hearing.",
        "gains": [1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "High-Frequency Loss (Presbycusis)": {
        "description": "Age-related hearing loss. Boosts 2 kHz-8 kHz to compensate "
                       "for reduced sensitivity to high-frequency consonants (s, f, th).",
        "gains": [1.0, 1.0, 1.2, 1.6, 1.8],
    },
    "Mild Flat Loss": {
        "description": "Uniform mild hearing loss across all frequencies. "
                       "All bands boosted equally.",
        "gains": [1.3, 1.3, 1.3, 1.3, 1.3],
    },
    "Cookie-Bite (Mid-Frequency Loss)": {
        "description": "Loss concentrated in the 500 Hz–2 kHz speech range. "
                       "Named for the U-shape on an audiogram.",
        "gains": [1.0, 1.4, 1.6, 1.4, 1.0],
    },
    "Noise-Induced Hearing Loss (NIHL)": {
        "description": "Characteristic notch around 4 kHz from prolonged noise exposure "
                       "(factory workers, musicians). Boosts high-mid aggressively.",
        "gains": [1.0, 1.0, 1.1, 1.8, 1.5],
    },
    "Reverse Slope (Low-Frequency Loss)": {
        "description": "Rare pattern where low frequencies are affected. "
                       "Boosts bass and low-mid to restore vowel clarity.",
        "gains": [1.7, 1.5, 1.1, 1.0, 1.0],
    },
}


def apply_equalizer(data, fs, gains):
    """
    data: 1D numpy array (audio signal)
    fs: sample rate
    gains: list of 5 gain multipliers, e.g. [1.0, 1.2, 0.8, 1.0, 1.5]
           1.0 = no change, >1.0 = boost, <1.0 = cut
    """
    if len(gains) != len(BANDS):
        raise ValueError(f"Expected {len(BANDS)} gains, got {len(gains)}")

    output = np.zeros_like(data, dtype=np.float64)
    nyquist = fs / 2

    for (low, high), gain in zip(BANDS, gains):
        high_clamped = min(high, nyquist - 1)  # avoid filter design errors near Nyquist
        if low >= high_clamped:
            continue
        b, a = butter_bandpass(low, high_clamped, fs, order=4)
        band_signal = apply_filter(data, b, a)
        output += gain * band_signal

    return output


def get_profile_gains(profile_name):
    """Return gains list for a named hearing profile."""
    if profile_name not in HEARING_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. "
                         f"Available: {list(HEARING_PROFILES.keys())}")
    return HEARING_PROFILES[profile_name]["gains"]
