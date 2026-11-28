"""
Feature extraction for AlertNet (Member 1).

Techniques:
  * FFT              -> fft_spectrum(), dominant_frequency()
  * Spectrogram      -> spectrogram()
  * Band features    -> band_matrix(): spectrogram grouped into log-spaced
                        frequency bands (like a simplified mel scale)
  * Cross-correlation-> normalized_cross_correlation()
  * RMS energy       -> see audio_preprocessing.rms / rms_db

The fingerprint used for classification (extract_features) has two parts:
  1. mean band profile  : WHICH frequencies the sound contains
  2. band variation     : HOW those frequencies change over time
                          (beeping, sweeping siren, two-tone alternation)
Both parts are z-normalized, so loudness does not matter.
"""

import numpy as np
from scipy.signal import stft

from detection.audio_preprocessing import BAND_LOW_HZ, BAND_HIGH_HZ

N_FFT = 1024
HOP = 256
N_BANDS = 48
COMPRESSION = 0.5


# ---------------------------------------------------------------- FFT ----
def fft_spectrum(audio, sr):
    """Magnitude spectrum of the whole signal. Returns (freqs, magnitude)."""
    windowed = audio * np.hanning(len(audio))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    return freqs, mag


def dominant_frequency(audio, sr, fmin=BAND_LOW_HZ, fmax=BAND_HIGH_HZ):
    """Frequency (Hz) with the strongest magnitude inside [fmin, fmax]."""
    freqs, mag = fft_spectrum(audio, sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(mag[mask])])


# -------------------------------------------------------- Spectrogram ----
def spectrogram(audio, sr, n_fft=N_FFT, hop=HOP):
    """Power spectrogram. Returns (freqs, times, power[freq, time])."""
    freqs, times, z = stft(audio, fs=sr, nperseg=n_fft,
                           noverlap=n_fft - hop, boundary=None, padded=False)
    return freqs, times, np.abs(z) ** 2


def _band_edges(sr, n_bands=N_BANDS, fmin=BAND_LOW_HZ, fmax=BAND_HIGH_HZ):
    fmax = min(fmax, sr / 2 * 0.95)
    return np.geomspace(fmin, fmax, n_bands + 1)


def band_matrix(audio, sr, n_bands=N_BANDS):
    """Spectrogram grouped into log-spaced bands, in dB. Shape (bands, frames)."""
    freqs, _, power = spectrogram(audio, sr)
    edges = _band_edges(sr, n_bands)
    bands = np.zeros((n_bands, power.shape[1]), dtype=np.float64)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if not mask.any():  # very narrow low band: take nearest bin
            mask = np.zeros_like(freqs, dtype=bool)
            mask[np.argmin(np.abs(freqs - edges[i]))] = True
        bands[i] = power[mask].sum(axis=0)   # sum: pink noise -> flat
    return 10.0 * np.log10(bands + 1e-12)


# ------------------------------------------------- Cross-correlation ----
def normalized_cross_correlation(a, b):
    """
    Maximum normalized cross-correlation between two 1-D signals
    (value in [-1, 1]; 1 = identical shape at some time shift).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    corr = np.correlate(a, b, mode="full") / min(len(a), len(b))
    return float(np.max(corr))


def pearson(a, b):
    """Correlation coefficient of two feature vectors (shape similarity)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ------------------------------------------------------ Fingerprint ----
def _smooth_bands(m, kernel=(0.25, 0.5, 0.25)):
    k = np.asarray(kernel)[:, None]
    padded = np.pad(m, ((1, 1), (0, 0)), mode="edge")
    return k[0] * padded[:-2] + k[1] * padded[1:-1] + k[2] * padded[2:]


def _znorm(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def extract_features(audio, sr):
    """
    Feature vector for one analysis window.
    Returns dict with 'profile', 'variation' and 'tonality'.
    """
    bands_db = band_matrix(audio, sr)             # (bands, frames) in dB
    # Compressed linear power instead of dB: broadband background noise
    # adds an almost constant offset across bands, which the correlation
    # (pearson) ignores -> much more robust to noise than dB values.
    power = 10.0 ** (bands_db / 10.0)
    power = power / (power.mean() + 1e-12)        # loudness independent
    comp = power ** COMPRESSION
    # Light smoothing across neighbouring bands: a tone that is a few Hz
    # higher/lower (different speaker, recording) still matches.
    comp = _smooth_bands(comp)
    profile = comp.mean(axis=1)                   # spectral shape
    variation = comp.std(axis=1)                  # change over time
    # Tonality: how much the strongest band stands out (alarms are tonal,
    # noise / speech are much flatter).
    mean_db = bands_db.mean(axis=1)
    tonality = float(mean_db.max() - np.median(mean_db))
    return {
        "profile": _znorm(profile),
        "variation": _znorm(variation),
        "tonality": tonality,
    }


def feature_similarity(f1, f2, w_profile=0.65, w_variation=0.35):
    """Similarity in [0, 1] between two feature dicts."""
    s = (w_profile * pearson(f1["profile"], f2["profile"]) +
         w_variation * pearson(f1["variation"], f2["variation"]))
    return max(0.0, min(1.0, s))
