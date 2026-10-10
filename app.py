"""
app.py — Digital Hearing Aid Simulation Pipeline
Load noisy speech → suppress background noise (notch + low-pass)
→ apply hearing-loss-compensating EQ → analyze → save → log to SQLite.
"""

from src.io_utils import load_audio, save_audio
from src.filters import (
    butter_lowpass, apply_filter, notch_filter, frequency_response
)
from src.equalizer import apply_equalizer, HEARING_PROFILES, get_profile_gains
from src.analysis import compute_fft, compute_spectrogram, calculate_snr, hum_reduction_db
from src.visualize import plot_waveform, plot_fft, plot_spectrogram, plot_frequency_response
from src.db import init_db, log_result


def main():
    init_db()

    raw_path = "data/raw/noisy_sample.wav"
    clean_ref_path = "data/raw/clean_reference.wav"   # optional, only exists for synthetic test audio
    out_path = "data/processed/cleaned_sample.wav"

    # ---- Select Hearing Profile ----
    profile_name = "High-Frequency Loss (Presbycusis)"  # most common case
    gains = get_profile_gains(profile_name)
    profile_info = HEARING_PROFILES[profile_name]

    print(f"=== Digital Hearing Aid Simulation ===")
    print(f"Hearing Profile: {profile_name}")
    print(f"  -> {profile_info['description']}")
    print(f"  -> EQ Gains: {dict(zip(['Bass','Low-Mid','Mid','High-Mid','Treble'], gains))}")
    print()

    # ---- 1. Load ----
    data, fs = load_audio(raw_path)
    print(f"Loaded '{raw_path}': {fs} Hz, {len(data)/fs:.2f} s")

    # ---- 2. Visualize original (noisy input as patient would hear) ----
    plot_waveform(data, fs, "Input Signal (Noisy Environment)")
    freqs, mag = compute_fft(data, fs)
    plot_fft(freqs, mag, "Input Spectrum — Before Hearing Aid Processing")
    f, t_spec, Sxx = compute_spectrogram(data, fs)
    plot_spectrogram(f, t_spec, Sxx, "Input Spectrogram — Noisy Environment")

    # ---- 3. Show noise suppression filter response ----
    b_lp, a_lp = butter_lowpass(cutoff=4000, fs=fs, order=4)
    w, h_db = frequency_response(b_lp, a_lp, fs)
    plot_frequency_response(w, h_db, "Noise Suppression Filter Response (LP cutoff=4 kHz)")

    # ---- 4. Noise Suppression: remove electrical hum + high-freq hiss ----
    print("Applying noise suppression (50 Hz notch + 4 kHz low-pass)...")
    denoised = notch_filter(data, fs, freq=50.0, quality=30.0)     # remove mains hum
    denoised = notch_filter(denoised, fs, freq=100.0, quality=30.0) # remove 2nd harmonic
    denoised = apply_filter(denoised, b_lp, a_lp)                   # remove high-freq hiss

    # ---- 5. Hearing-Loss Compensation (EQ based on audiogram profile) ----
    print(f"Applying hearing compensation: {profile_name}...")
    processed = apply_equalizer(denoised, fs, gains)

    # ---- 6. Visualize output (what the patient hears through the aid) ----
    plot_waveform(processed, fs, "Output Signal — After Hearing Aid Processing")
    freqs2, mag2 = compute_fft(processed, fs)
    plot_fft(freqs2, mag2, "Output Spectrum — Hearing Aid Output")

    # ---- 7. Save output ----
    save_audio(out_path, processed, fs)
    print(f"Saved processed audio to '{out_path}'")

    # ---- 8. Quality Metrics ----
    try:
        clean_ref, _ = load_audio(clean_ref_path)
        snr_before = calculate_snr(clean_ref, data)
        snr_after = calculate_snr(clean_ref, processed)
    except FileNotFoundError:
        # no clean reference available -- use hum-reduction proxy
        snr_before = 0.0
        snr_after = hum_reduction_db(data, processed, fs, hum_freq=50.0)

    print(f"\nSNR/quality before hearing aid: {snr_before:.2f} dB")
    print(f"SNR/quality after hearing aid:  {snr_after:.2f} dB")
    print(f"Improvement:                    {snr_after - snr_before:.2f} dB")

    # ---- 9. Log to SQLite ----
    log_result(
        filename="noisy_sample.wav",
        samplerate=fs,
        filter_type=f"notch50+notch100+lp4000+eq_{profile_name}",
        cutoff_low=50.0,
        cutoff_high=4000.0,
        eq_gains=gains,
        snr_before=snr_before,
        snr_after=snr_after,
    )
    print("Logged results to database/project.db")


if __name__ == "__main__":
    main()
