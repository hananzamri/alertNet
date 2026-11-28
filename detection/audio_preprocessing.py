"""
Audio preprocessing for AlertNet (Member 1).

stereo -> mono, resampling, normalization, simple noise reduction
(band-pass filter), windowing and RMS energy.
"""

from math import gcd

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly

# Alarm sounds live roughly between 300 Hz and 5 kHz.
BAND_LOW_HZ = 250
BAND_HIGH_HZ = 6000


def pcm_to_float(data):
    """Convert int16/int32/uint8/float WAV data to float32 in [-1, 1]."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def to_mono(audio):
    """Average channels: shape (n, ch) -> (n,)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def resample(audio, orig_sr, target_sr):
    """Polyphase resampling to the standard sample rate."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def normalize(audio, target_peak=0.9, eps=1e-9):
    """Peak normalization (volume-independent comparison)."""
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak < eps:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def bandpass(audio, sr, low=BAND_LOW_HZ, high=BAND_HIGH_HZ, order=4):
    """Remove rumble / hiss outside the alarm frequency range."""
    high = min(high, sr / 2 * 0.95)
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def rms(audio):
    """Root-mean-square energy."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def rms_db(audio):
    """RMS in dBFS (0 dB = full scale sine peak)."""
    return 20.0 * np.log10(rms(audio) + 1e-12)


def frame_signal(audio, sr, window_seconds=1.0, hop_seconds=0.25):
    """Split audio into overlapping windows. Returns array (n_windows, win)."""
    win = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    if len(audio) < win:
        return np.empty((0, win), dtype=np.float32)
    n = 1 + (len(audio) - win) // hop
    return np.stack([audio[i * hop:i * hop + win] for i in range(n)])


def preprocess(audio, sr):
    """Full chain used before feature extraction."""
    audio = to_mono(audio)
    audio = bandpass(audio, sr)
    return audio
