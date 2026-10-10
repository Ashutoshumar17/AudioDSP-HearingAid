"""
io_utils.py
Handles loading and saving WAV audio files.
"""

import soundfile as sf
import numpy as np


def load_audio(filepath):
    """
    Load a WAV file.
    Returns: (data, samplerate)
    - data: 1D numpy array (mono). Stereo files are averaged to mono.
    """
    data, samplerate = sf.read(filepath)
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    return data.astype(np.float64), samplerate


def save_audio(filepath, data, samplerate):
    """Save a 1D numpy array as a WAV file."""
    # clip to avoid overflow/clipping artifacts on save
    data = np.clip(data, -1.0, 1.0)
    sf.write(filepath, data, samplerate)
