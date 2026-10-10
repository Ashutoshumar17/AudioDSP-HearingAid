"""
streamlit_app.py — Digital Hearing Aid Interactive Simulator
Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
from src.io_utils import load_audio, save_audio
from src.filters import notch_filter, butter_lowpass, apply_filter
from src.equalizer import apply_equalizer, BAND_NAMES, HEARING_PROFILES, get_profile_gains
from src.analysis import compute_fft, calculate_snr, hum_reduction_db

st.set_page_config(page_title="Digital Hearing Aid Simulator", page_icon="🦻", layout="wide")

st.title("🦻 Digital Hearing Aid Simulator")
st.markdown(
    "Upload an audio file or use the test sample to simulate how a digital hearing aid "
    "processes sound: **noise suppression** → **hearing-loss compensation** → **clean output**."
)

# ---- Sidebar: Hearing Profile Selection ----
st.sidebar.header("👤 Patient Hearing Profile")

profile_names = list(HEARING_PROFILES.keys())
selected_profile = st.sidebar.selectbox(
    "Select hearing loss type",
    profile_names,
    index=1,  # default to Presbycusis
)

profile = HEARING_PROFILES[selected_profile]
st.sidebar.info(f"**{selected_profile}**\n\n{profile['description']}")

use_custom = st.sidebar.checkbox("Override with custom EQ gains")

if use_custom:
    st.sidebar.subheader("Custom EQ Gains")
    gains = []
    for name, default in zip(BAND_NAMES, profile["gains"]):
        gains.append(st.sidebar.slider(name, 0.0, 2.5, default, 0.1))
else:
    gains = profile["gains"]

# Display current gains
st.sidebar.subheader("Active Gains")
for name, g in zip(BAND_NAMES, gains):
    bar = "█" * int(g * 10)
    st.sidebar.text(f"{name:>10}: {g:.1f}  {bar}")

# ---- Sidebar: Noise Suppression Settings ----
st.sidebar.header("🔇 Noise Suppression")
remove_hum = st.sidebar.checkbox("Remove electrical hum", value=True)
hum_freq = st.sidebar.selectbox("Mains frequency", [50.0, 60.0]) if remove_hum else None
remove_harmonics = st.sidebar.checkbox("Also remove 2nd harmonic", value=True) if remove_hum else False
apply_lowpass = st.sidebar.checkbox("Low-pass filter (remove hiss)", value=True)
lp_cutoff = st.sidebar.slider("Low-pass cutoff (Hz)", 2000, 8000, 4000, 100) if apply_lowpass else None

# ---- Main Area: Audio Upload ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Audio")
    use_test = st.checkbox("Use built-in test sample", value=True)

    if use_test:
        input_path = "data/raw/noisy_sample.wav"
        try:
            data, fs = load_audio(input_path)
            st.audio(input_path)
            st.caption(f"Sample rate: {fs} Hz | Duration: {len(data)/fs:.2f}s")
        except FileNotFoundError:
            st.error("Test sample not found. Run `python generate_test_audio.py` first.")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded:
            input_path = "data/raw/temp_upload.wav"
            with open(input_path, "wb") as f:
                f.write(uploaded.read())
            data, fs = load_audio(input_path)
            st.audio(input_path)
            st.caption(f"Sample rate: {fs} Hz | Duration: {len(data)/fs:.2f}s")
        else:
            st.info("Upload a WAV file or check 'Use built-in test sample'.")
            st.stop()

# ---- Process Button ----
if st.button("🦻 Process Through Hearing Aid", type="primary"):
    signal = data.copy()

    # Step 1: Noise Suppression
    with st.spinner("Suppressing background noise..."):
        if remove_hum:
            signal = notch_filter(signal, fs, freq=hum_freq, quality=30.0)
            if remove_harmonics:
                signal = notch_filter(signal, fs, freq=hum_freq * 2, quality=30.0)
        if apply_lowpass:
            b_lp, a_lp = butter_lowpass(cutoff=lp_cutoff, fs=fs, order=4)
            signal = apply_filter(signal, b_lp, a_lp)

    # Step 2: Hearing-Loss Compensation
    with st.spinner(f"Applying hearing compensation ({selected_profile})..."):
        processed = apply_equalizer(signal, fs, gains)

    # Save output
    out_path = "data/processed/hearing_aid_output.wav"
    save_audio(out_path, processed, fs)

    with col2:
        st.subheader("📤 Hearing Aid Output")
        st.audio(out_path)
        st.success(f"Profile: {selected_profile}")

    # ---- Metrics ----
    st.subheader("📊 Quality Metrics")
    try:
        clean_ref, _ = load_audio("data/raw/clean_reference.wav")
        snr_before = calculate_snr(clean_ref, data)
        snr_after = calculate_snr(clean_ref, processed)
    except FileNotFoundError:
        snr_before = 0.0
        snr_after = hum_reduction_db(data, processed, fs, hum_freq=50.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("SNR Before", f"{snr_before:.2f} dB")
    m2.metric("SNR After", f"{snr_after:.2f} dB")
    m3.metric("Improvement", f"{snr_after - snr_before:.2f} dB", delta=f"{snr_after - snr_before:.1f} dB")

    # ---- Spectrum Comparison ----
    st.subheader("📈 Spectrum Comparison")
    import matplotlib.pyplot as plt
    from src.analysis import compute_fft

    freqs_b, mag_b = compute_fft(data, fs)
    freqs_a, mag_a = compute_fft(processed, fs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs_b, mag_b, alpha=0.6, label="Before (noisy input)", color="red")
    ax.plot(freqs_a, mag_a, alpha=0.8, label="After (hearing aid output)", color="green")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Spectrum: Before vs After Hearing Aid Processing")
    ax.legend()
    ax.set_xlim(0, fs / 2)
    plt.tight_layout()
    st.pyplot(fig)
