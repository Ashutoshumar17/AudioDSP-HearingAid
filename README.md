# 🦻 Digital Hearing Aid Application Using Noise Reduction Filter

**Mini Project (CHAMP) — Digital Signal Processing**

A Python-based simulation of a digital hearing aid system that demonstrates core DSP concepts: **noise suppression** (notch filtering, low-pass filtering) and **hearing-loss compensation** (multi-band equalization based on clinical audiogram profiles).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Theory & DSP Concepts](#theory--dsp-concepts)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Hearing Loss Profiles](#hearing-loss-profiles)
- [How to Run](#how-to-run)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Results & Analysis](#results--analysis)
- [Technologies Used](#technologies-used)
- [References](#references)

---

## Project Overview

A digital hearing aid is a miniature computer that processes sound in real time to help people with hearing loss hear more clearly. The core DSP operations inside a hearing aid are:

1. **Noise Suppression** — Remove unwanted background noise (electrical hum, high-frequency hiss, environmental noise) using digital filters.
2. **Hearing-Loss Compensation** — Boost specific frequency bands where the patient has reduced hearing sensitivity, based on their audiogram (hearing test results).

This project implements both of these operations in Python using standard DSP libraries (`scipy.signal`, `numpy`) and provides:
- A **command-line pipeline** (`app.py`) for batch processing
- An **interactive web UI** (`streamlit_app.py`) for real-time experimentation

---

## Objectives

1. Design and implement IIR digital filters (Butterworth low-pass, notch) for noise suppression.
2. Build a 5-band graphic equalizer to simulate hearing-loss compensation.
3. Implement clinical hearing-loss profiles (Presbycusis, NIHL, Cookie-Bite, etc.) based on standard audiogram patterns.
4. Compute signal quality metrics (SNR) to quantify the improvement.
5. Visualize signals in both time domain (waveform) and frequency domain (FFT, spectrogram).
6. Log all experiments to a SQLite database for reproducibility.

---

## Theory & DSP Concepts

### 1. IIR Notch Filter (Hum Removal)

Electrical hum at 50/60 Hz (from power lines) is a common noise source in audio recordings and hearing aid microphones. A **notch filter** (band-reject filter) removes a very narrow frequency band while leaving everything else untouched.

**Transfer function:**

```
        b₀ + b₁z⁻¹ + b₂z⁻²
H(z) = ———————————————————————
        1 + a₁z⁻¹ + a₂z⁻²
```

- The **quality factor Q** controls the notch width: higher Q = narrower notch.
- We use Q = 30 for a sharp notch that removes 50 Hz hum without affecting nearby speech frequencies.
- We also remove the **2nd harmonic** (100 Hz) for thorough hum suppression.

**Implementation:** `scipy.signal.iirnotch()` → applied with `filtfilt()` for zero-phase filtering.

### 2. Butterworth Low-Pass Filter (Hiss Removal)

High-frequency hiss (broadband noise above ~4 kHz) is suppressed using a **Butterworth low-pass filter**:

- Butterworth filters have a **maximally flat passband** — no ripples in the frequencies we want to keep.
- We use a 4th-order filter with cutoff at 4 kHz. This preserves all speech content (speech energy is concentrated below 4 kHz) while removing high-frequency noise.

**Design equation:**

```
|H(jω)|² = 1 / (1 + (ω/ωc)^(2N))
```

where N = filter order and ωc = cutoff frequency.

### 3. Zero-Phase Filtering (filtfilt)

Standard IIR filters (`lfilter`) introduce **phase distortion** — different frequencies are delayed by different amounts, smearing the signal. We use `scipy.signal.filtfilt()` which:
1. Filters the signal forward
2. Filters the result backward
3. The phase shifts cancel out → **zero phase distortion**

This is critical for hearing aids because phase distortion degrades speech intelligibility.

### 4. Multi-Band Graphic Equalizer

A graphic equalizer divides the audio spectrum into bands and applies independent gain to each band:

| Band | Frequency Range | Speech Content |
|------|----------------|----------------|
| Bass | 20 – 250 Hz | Fundamental voice pitch, vowel warmth |
| Low-Mid | 250 – 500 Hz | Vowel body, nasal sounds |
| Mid | 500 – 2000 Hz | Core speech intelligibility |
| High-Mid | 2000 – 6000 Hz | Consonant clarity (s, f, t, th) |
| Treble | 6000 – 20000 Hz | Sibilance, air, brightness |

**Implementation:** Each band is isolated using a Butterworth band-pass filter, multiplied by its gain, and summed back together.

### 5. FFT (Fast Fourier Transform)

The FFT converts a time-domain signal into its frequency components, allowing us to:
- **Visualize** which frequencies are present (spectrum plot)
- **Measure** noise energy in specific frequency bands
- **Verify** that our filters are working correctly

### 6. SNR (Signal-to-Noise Ratio)

```
SNR (dB) = 10 × log₁₀(P_signal / P_noise)
```

- Higher SNR = cleaner signal
- If a clean reference is available: `noise = noisy_signal - clean_reference`
- If no clean reference: we use **hum reduction in dB** as a proxy metric

### 7. Spectrogram (Time-Frequency Analysis)

A spectrogram shows how the frequency content of a signal changes over time. It's computed using the Short-Time Fourier Transform (STFT):
1. Divide the signal into overlapping windows (Hamming window, 1024 samples)
2. Compute FFT for each window
3. Plot the magnitude as a color map (time × frequency × power)

---

## System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │         Digital Hearing Aid System          │
                    └─────────────────────────────────────────────┘

   ┌──────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────┐
   │  Input   │────▶│  Noise Suppression   │────▶│  Hearing-Loss        │────▶│  Output  │
   │  Audio   │     │  Stage               │     │  Compensation (EQ)   │     │  Audio   │
   │ (noisy)  │     │                      │     │                      │     │ (clean)  │
   └──────────┘     │  • 50 Hz Notch       │     │  • 5-Band EQ         │     └──────────┘
                    │  • 100 Hz Notch      │     │  • Audiogram Presets  │
                    │  • 4 kHz Low-Pass    │     │  • Custom Gains      │
                    └──────────────────────┘     └──────────────────────┘
                              │                            │
                              ▼                            ▼
                    ┌──────────────────────────────────────────────┐
                    │              Analysis & Logging              │
                    │  • FFT Spectrum  • Spectrogram  • SNR       │
                    │  • Waveform     • SQLite Log    • Plots     │
                    └──────────────────────────────────────────────┘
```

---

## Project Structure

```
dsp_equalizer_project/
├── data/
│   ├── raw/                       # Input audio (noisy_sample.wav)
│   └── processed/                 # Cleaned output audio
├── src/
│   ├── __init__.py
│   ├── io_utils.py                # Load/save WAV files
│   ├── filters.py                 # Low-pass, high-pass, band-pass, notch filter design
│   ├── equalizer.py               # 5-band graphic EQ + hearing loss presets
│   ├── analysis.py                # FFT, spectrogram, SNR calculation
│   ├── visualize.py               # Waveform / spectrum / spectrogram plots
│   └── db.py                      # SQLite experiment logging
├── database/
│   └── project.db                 # Auto-created on first run
├── app.py                         # Main pipeline (run this first)
├── generate_test_audio.py         # Creates synthetic hearing-aid test audio
├── streamlit_app.py               # Interactive web UI
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Module Documentation

### `src/io_utils.py` — Audio I/O

| Function | Parameters | Description |
|----------|-----------|-------------|
| `load_audio(filepath)` | Path to WAV file | Loads WAV, converts stereo→mono, returns `(data, samplerate)` |
| `save_audio(filepath, data, samplerate)` | Path, 1D array, sample rate | Saves audio, clips to [-1, 1] to prevent overflow |

### `src/filters.py` — Digital Filter Design

| Function | Parameters | Description |
|----------|-----------|-------------|
| `butter_lowpass(cutoff, fs, order)` | Cutoff Hz, sample rate, order | Returns Butterworth LP filter coefficients (b, a) |
| `butter_highpass(cutoff, fs, order)` | Cutoff Hz, sample rate, order | Returns Butterworth HP filter coefficients (b, a) |
| `butter_bandpass(lowcut, highcut, fs, order)` | Low/high Hz, sample rate, order | Returns Butterworth BP filter coefficients (b, a) |
| `apply_filter(data, b, a)` | Signal, filter coeffs | Applies zero-phase filtering using `filtfilt` |
| `notch_filter(data, fs, freq, quality)` | Signal, sample rate, notch freq, Q | Removes a narrow frequency band (e.g., 50 Hz hum) |
| `frequency_response(b, a, fs, worN)` | Filter coeffs, sample rate, num points | Returns (freqs_hz, magnitude_dB) for Bode plot |

### `src/equalizer.py` — Graphic Equalizer + Hearing Profiles

| Function | Parameters | Description |
|----------|-----------|-------------|
| `apply_equalizer(data, fs, gains)` | Signal, sample rate, 5 gain values | Applies 5-band EQ with given gain multipliers |
| `get_profile_gains(profile_name)` | Profile name string | Returns gains list for a clinical hearing profile |

**Constants:** `BANDS`, `BAND_NAMES`, `HEARING_PROFILES`

### `src/analysis.py` — Signal Analysis

| Function | Parameters | Description |
|----------|-----------|-------------|
| `compute_fft(data, fs)` | Signal, sample rate | Returns one-sided (freqs, magnitude) |
| `compute_spectrogram(data, fs)` | Signal, sample rate | Returns (f, t, Sxx) for spectrogram plot |
| `band_energy(data, fs, low_hz, high_hz)` | Signal, sample rate, freq range | Computes energy in a specific frequency band |
| `calculate_snr(clean_ref, noisy)` | Clean reference, noisy signal | True SNR in dB (requires clean reference) |
| `hum_reduction_db(before, after, fs, hum_freq, bandwidth)` | Before/after signals, hum freq | Proxy metric: energy reduction around hum frequency |

### `src/visualize.py` — Plotting

| Function | Parameters | Description |
|----------|-----------|-------------|
| `plot_waveform(data, fs, title)` | Signal, sample rate, title | Time-domain amplitude plot |
| `plot_fft(freqs, magnitude, title)` | Freq array, mag array, title | Frequency spectrum plot |
| `plot_spectrogram(f, t, Sxx, title)` | Freq, time, power arrays, title | Time-frequency heatmap |
| `plot_frequency_response(freqs_hz, magnitude_db, title)` | Freq, dB arrays, title | Filter Bode magnitude plot |

### `src/db.py` — SQLite Database

| Function | Parameters | Description |
|----------|-----------|-------------|
| `init_db(db_path)` | DB file path | Creates `processed_files` table if not exists |
| `log_result(...)` | Filename, sample rate, filter settings, SNR values | Inserts a processing experiment record |
| `query_results(db_path)` | DB file path | Returns all logged experiments (newest first) |
| `query_best_improvements(db_path, min_improvement)` | DB path, threshold | Returns experiments ranked by SNR improvement |

---

## Hearing Loss Profiles

The equalizer includes 6 clinical hearing-loss presets based on standard audiogram patterns:

| Profile | Bass | Low-Mid | Mid | High-Mid | Treble | Description |
|---------|------|---------|-----|----------|--------|-------------|
| Normal Hearing | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | No amplification needed |
| **High-Frequency Loss (Presbycusis)** | 1.0 | 1.0 | 1.2 | 1.6 | 1.8 | Age-related loss; most common type |
| Mild Flat Loss | 1.3 | 1.3 | 1.3 | 1.3 | 1.3 | Uniform loss across all frequencies |
| Cookie-Bite (Mid-Freq Loss) | 1.0 | 1.4 | 1.6 | 1.4 | 1.0 | U-shaped loss in speech range |
| Noise-Induced (NIHL) | 1.0 | 1.0 | 1.1 | 1.8 | 1.5 | 4 kHz notch from noise exposure |
| Reverse Slope (Low-Freq Loss) | 1.7 | 1.5 | 1.1 | 1.0 | 1.0 | Rare; low frequencies affected |

---

## How to Run

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Generate Test Audio

```bash
python generate_test_audio.py
```

This creates:
- `data/raw/noisy_sample.wav` — Simulated speech + electrical hum + crowd noise + hiss
- `data/raw/clean_reference.wav` — Clean speech reference (for SNR calculation)

### Step 2: Run the Pipeline

```bash
python app.py
```

### Step 3 (Optional): Interactive Web UI

```bash
streamlit run streamlit_app.py
```

This launches a browser-based hearing aid simulator where you can:
- Upload your own audio files
- Select different hearing loss profiles
- Adjust noise suppression settings
- Compare before/after audio playback
- View spectrum comparison plots

---

## Pipeline Walkthrough

When you run `python app.py`, the following happens:

1. **Load** `data/raw/noisy_sample.wav`
2. **Visualize** the original noisy signal (waveform, FFT spectrum, spectrogram)
3. **Display** the noise suppression filter's frequency response (Bode plot)
4. **Noise Suppression:**
   - Remove 50 Hz electrical hum (IIR notch filter, Q=30)
   - Remove 100 Hz harmonic (IIR notch filter, Q=30)
   - Remove high-frequency hiss (4th-order Butterworth LP, cutoff=4 kHz)
5. **Hearing Compensation:** Apply 5-band equalizer with Presbycusis preset
6. **Visualize** the processed output (waveform, FFT spectrum)
7. **Save** cleaned audio to `data/processed/cleaned_sample.wav`
8. **Compute** SNR before and after processing
9. **Log** all settings and metrics to `database/project.db` (SQLite)

---

## Results & Analysis

### Test Signal

The synthetic test audio simulates a real hearing-aid scenario:
- **Speech component:** Formant-based vowel synthesis (F0 ~150 Hz, F1 ~700 Hz, F2 ~1200 Hz, F3 ~2500 Hz) with amplitude modulation to simulate syllable-like bursts
- **Noise components:** 50 Hz mains hum + harmonics, crowd babble noise (200–4000 Hz), broadband hiss

### Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| SNR | ~9–10 dB | ~13–15 dB | **+3 to +5 dB improvement** |
| 50 Hz Hum Energy | High | Near zero | Notch filter effective |
| High-freq Hiss | Present | Suppressed | LP filter effective |
| Speech Clarity | Poor | Enhanced | EQ boosted speech bands |

### Key Observations

- **Zero-phase filtering** (`filtfilt`) preserves speech timing and intelligibility — critical for hearing aids
- **Notch filter** removes hum precisely without affecting nearby speech content
- **Presbycusis profile** boosts high-mid (2–6 kHz) where consonant sounds (s, f, th) live — improving word discrimination
- **Nyquist clamping** in the equalizer prevents band-pass filter design errors at high sample rates

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.x | Primary language |
| NumPy | Array operations, FFT |
| SciPy (`scipy.signal`) | Filter design (Butterworth, IIR notch), `filtfilt`, spectrogram |
| Matplotlib | Waveform, spectrum, spectrogram, Bode plots |
| SoundFile | WAV file I/O |
| Streamlit | Interactive web UI |
| SQLite3 | Experiment logging |

---

## References

1. Oppenheim, A. V., & Schafer, R. W. — *Discrete-Time Signal Processing*, Prentice Hall.
2. Haykin, S. — *Adaptive Filter Theory*, Pearson.
3. WHO — *World Report on Hearing* (2021).
4. SciPy Documentation — [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
5. Dillon, H. — *Hearing Aids*, Thieme (standard hearing aid engineering textbook).
6. ASHA — Audiogram patterns and hearing loss types: [www.asha.org](https://www.asha.org)
