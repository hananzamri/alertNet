"""
First coding task (Member 1): understand the reference alarm sounds.

    python tools/inspect_audio.py                        # all 3 references
    python tools/inspect_audio.py sounds/test/x.wav      # any WAV file

For each file prints sample rate, duration, RMS level and dominant
frequency, and saves a PNG with waveform, FFT spectrum and spectrogram
to sounds/plots/.
"""

import os
import sys

import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from detection.audio_capture import load_wav  # noqa: E402
from detection.audio_preprocessing import rms_db  # noqa: E402
from detection.feature_extraction import (  # noqa: E402
    fft_spectrum, dominant_frequency, spectrogram)

PLOT_DIR = os.path.join(ROOT, "sounds", "plots")


def inspect(path):
    original_sr, raw = wavfile.read(path)
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    audio, sr = load_wav(path)
    duration = len(audio) / sr

    print(f"\n{os.path.basename(path)}")
    print(f"  original sample rate : {original_sr} Hz, {channels} channel(s)")
    print(f"  standardized to      : {sr} Hz mono")
    print(f"  duration             : {duration:.2f} s")
    print(f"  RMS level            : {rms_db(audio):.1f} dBFS")
    print(f"  dominant frequency   : {dominant_frequency(audio, sr):.0f} Hz")

    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    t = np.arange(len(audio)) / sr
    ax[0].plot(t, audio, linewidth=0.5)
    ax[0].set(title=f"Waveform – {os.path.basename(path)}",
              xlabel="Time (s)", ylabel="Amplitude")

    freqs, mag = fft_spectrum(audio, sr)
    ax[1].plot(freqs, 20 * np.log10(mag + 1e-9), linewidth=0.6)
    ax[1].set(title="FFT spectrum", xlabel="Frequency (Hz)",
              ylabel="Magnitude (dB)", xlim=(0, 8000))

    f, tt, p = spectrogram(audio, sr)
    ax[2].pcolormesh(tt, f, 10 * np.log10(p + 1e-12), shading="auto")
    ax[2].set(title="Spectrogram", xlabel="Time (s)",
              ylabel="Frequency (Hz)", ylim=(0, 8000))
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, os.path.basename(path).replace(".wav", ".png"))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  plot saved           : {out}")


if __name__ == "__main__":
    files = sys.argv[1:] or [
        os.path.join(ROOT, "sounds", "reference", n)
        for n in ("fire_alarm.wav", "security_alarm.wav", "general_warning.wav")
    ]
    for p in files:
        inspect(p)
