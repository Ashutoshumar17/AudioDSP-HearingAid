"""
filters.py
Digital filter design and application using Butterworth filters
and an IIR notch filter (for hum removal).
"""

from scipy.signal import butter, filtfilt, iirnotch, freqz
import numpy as np


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def apply_filter(data, b, a):
    """Zero-phase filtering (no time-domain delay/distortion)."""
    return filtfilt(b, a, data)


def notch_filter(data, fs, freq=60.0, quality=30.0):
    """Remove a narrow frequency band (e.g. 50/60 Hz electrical hum)."""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)


def frequency_response(b, a, fs, worN=2000):
    """
    Returns (freqs_hz, magnitude_db) for plotting a filter's
    frequency response curve (Bode-style magnitude plot).
    """
    w, h = freqz(b, a, worN=worN, fs=fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    return w, magnitude_db
