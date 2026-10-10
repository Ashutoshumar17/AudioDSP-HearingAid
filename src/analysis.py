"""
analysis.py
FFT, spectrogram, and SNR utilities.
"""

import numpy as np
from scipy.signal import spectrogram


def compute_fft(data, fs):
    """Returns (freqs, magnitude) for the one-sided spectrum."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(np.fft.rfft(data))
    return freqs, magnitude


def compute_spectrogram(data, fs):
    f, t, Sxx = spectrogram(data, fs, window='hamming', nperseg=1024, noverlap=512)
    return f, t, Sxx


def band_energy(data, fs, low_hz, high_hz):
    """Energy of the signal within a specific frequency band (used as a noise proxy)."""
    freqs, magnitude = compute_fft(data, fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(magnitude[mask] ** 2))


def calculate_snr(clean_ref, noisy):
    """
    True SNR (needs a matching clean reference signal of the same length).
    """
    n = min(len(clean_ref), len(noisy))
    clean_ref, noisy = clean_ref[:n], noisy[:n]
    noise = noisy - clean_ref
    signal_power = np.mean(clean_ref ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def hum_reduction_db(before, after, fs, hum_freq=60.0, bandwidth=5.0):
    """
    Proxy metric when you DON'T have a clean reference:
    compares energy around the hum frequency before vs after filtering.
    """
    low, high = hum_freq - bandwidth, hum_freq + bandwidth
    e_before = band_energy(before, fs, low, high)
    e_after = band_energy(after, fs, low, high)
    if e_after == 0:
        return float("inf")
    return 10 * np.log10(e_before / e_after)
