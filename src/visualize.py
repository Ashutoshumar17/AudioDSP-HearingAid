"""
visualize.py
Plotting helpers: waveform, FFT spectrum, spectrogram, filter frequency response.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(data, fs, title="Waveform"):
    t = np.arange(len(data)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, data, linewidth=0.7)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_fft(freqs, magnitude, title="Frequency Spectrum"):
    plt.figure(figsize=(10, 3))
    plt.plot(freqs, magnitude, linewidth=0.7)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()


def plot_spectrogram(f, t, Sxx, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()


def plot_frequency_response(freqs_hz, magnitude_db, title="Filter Frequency Response"):
    plt.figure(figsize=(10, 3))
    plt.plot(freqs_hz, magnitude_db)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
