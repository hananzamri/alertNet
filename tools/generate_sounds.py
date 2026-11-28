"""
Generate reference and test sounds for AlertNet (Member 1).

    python tools/generate_sounds.py

Creates:
  sounds/reference/fire_alarm.wav, security_alarm.wav, general_warning.wav
  sounds/test/*.wav  + sounds/test/labels.json

The reference alarms are synthesized so the project works out of the box.
You may replace them with real recordings (same file names, >= 3 s long);
then re-run only the test part:  python tools/generate_sounds.py --tests-only
(the test variants are made from whatever reference files exist).
"""

import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection.audio_capture import save_wav, load_wav, TARGET_SR  # noqa: E402

SR = TARGET_SR
REF_DIR = os.path.join(ROOT, "sounds", "reference")
TEST_DIR = os.path.join(ROOT, "sounds", "test")
rng = np.random.default_rng(42)


# ------------------------------------------------------------ helpers --
def t_axis(seconds):
    return np.arange(int(seconds * SR)) / SR


def tone_from_freq(freq):
    """Rich tone (fundamental + harmonics) following an instantaneous freq."""
    phase = 2 * np.pi * np.cumsum(freq) / SR
    out = np.zeros_like(phase)
    for k, amp in ((1, 1.0), (3, 0.35), (5, 0.15)):
        ok = (k * freq) < 0.45 * SR          # skip harmonics above Nyquist
        out += amp * np.sin(k * phase) * ok   # (they would alias)
    return out


def smooth_gate(gate, ms=5):
    k = max(1, int(SR * ms / 1000))
    return np.convolve(gate, np.ones(k) / k, mode="same")


def norm(x, peak=0.8):
    return x / (np.max(np.abs(x)) + 1e-9) * peak


# ------------------------------------------------------------- alarms --
def fire_alarm(seconds, jitter=0.0, offset=0.0):
    """Temporal-3 pattern: 3 x (0.5 s beep, 0.5 s pause), then 1.5 s pause."""
    t = t_axis(seconds) + offset
    f = 3100 * (1 + jitter)
    cycle = t % 4.0
    gate = ((cycle < 3.0) & ((cycle % 1.0) < 0.5)).astype(float)
    return norm(tone_from_freq(np.full_like(t, f)) * smooth_gate(gate))


def security_alarm(seconds, jitter=0.0, offset=0.0):
    """Wailing siren sweeping 650 -> 1450 -> 650 Hz every 2 s."""
    t = t_axis(seconds) + offset
    tri = 1 - np.abs(((t / 2.0) % 1.0) * 2 - 1)        # 0..1..0
    freq = (650 + 800 * tri) * (1 + jitter)
    return norm(tone_from_freq(freq))


def general_warning(seconds, jitter=0.0, offset=0.0):
    """Two-tone hi-lo: 960 Hz / 770 Hz alternating every 0.5 s."""
    t = t_axis(seconds) + offset
    hi = ((t % 1.0) < 0.5)
    freq = np.where(hi, 960.0, 770.0) * (1 + jitter)
    return norm(tone_from_freq(freq))


ALARMS = {
    "FIRE_ALARM": ("fire_alarm.wav", fire_alarm),
    "SECURITY_ALARM": ("security_alarm.wav", security_alarm),
    "GENERAL_WARNING": ("general_warning.wav", general_warning),
}


# ------------------------------------------------- background sounds --
def pink_noise(seconds):
    n = int(seconds * SR)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    return norm(np.fft.irfft(spec, n))


def speech_like(seconds):
    """Babble: gliding pitch + formant filters + syllable envelope."""
    t = t_axis(seconds)
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t) + 20 * rng.standard_normal(len(t)).cumsum() / np.sqrt(len(t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = sum(np.sin(k * phase) / k for k in range(1, 30))
    out = np.zeros_like(src)
    for lo, hi in [(300, 900), (900, 2200), (2200, 3200)]:
        sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        out += sosfilt(sos, src) * rng.uniform(0.4, 1.0)
    syll = 0.5 * (1 + np.sin(2 * np.pi * 4.3 * t + 3 * np.sin(2 * np.pi * 0.9 * t)))
    return norm(out * syll ** 2)


def music_like(seconds):
    """Piano-like chords changing every 0.6 s."""
    t = t_axis(seconds)
    out = np.zeros_like(t)
    note_len = 0.6
    for i in range(int(seconds / note_len) + 1):
        root = 220 * 2 ** (rng.integers(0, 12) / 12)
        start = int(i * note_len * SR)
        seg = t_axis(1.2)
        env = np.exp(-3 * seg)
        chord = sum(np.sin(2 * np.pi * root * r * h * seg) / h
                    for r in (1, 1.26, 1.5) for h in (1, 2, 3))
        end = min(len(out), start + len(seg))
        out[start:end] += (chord * env)[:end - start]
    return norm(out)


def foreign_beep(seconds):
    """A beep that is NOT one of our alarms (microwave-style 2 kHz)."""
    t = t_axis(seconds)
    gate = ((t % 1.0) < 0.3).astype(float)
    return norm(np.sin(2 * np.pi * 2000 * t) * smooth_gate(gate))


# ------------------------------------------------------ degradations --
def at_snr(signal, noise, snr_db):
    ps = np.mean(signal ** 2)
    pn = np.mean(noise ** 2) + 1e-12
    return signal + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def far_away(signal):
    """Simulate distance: high frequencies lost, quieter, room echo."""
    sos = butter(4, 3500, btype="low", fs=SR, output="sos")
    s = sosfilt(sos, signal) * 0.25
    out = s.copy()
    for delay_ms, gain in [(23, 0.45), (41, 0.3), (67, 0.2)]:
        d = int(SR * delay_ms / 1000)
        out[d:] += s[:-d] * gain
    return out


def place(alarm, pre=1.5, post=3.5, ambience=None, amb_level=0.02):
    """Pad alarm with background before/after. Returns (audio, onset_s)."""
    total = pre + len(alarm) / SR + post
    bg = (ambience(total) if ambience else pink_noise(total)) * amb_level
    out = bg.copy()
    start = int(pre * SR)
    out[start:start + len(alarm)] += alarm
    return out, pre


# -------------------------------------------------------------- main --
def make_references():
    os.makedirs(REF_DIR, exist_ok=True)
    for cat, (fname, fn) in ALARMS.items():
        save_wav(os.path.join(REF_DIR, fname), fn(8.0))
        print("reference:", fname)


def reference_audio(cat):
    fname = ALARMS[cat][0]
    audio, _ = load_wav(os.path.join(REF_DIR, fname), SR)
    return audio


def make_tests():
    os.makedirs(TEST_DIR, exist_ok=True)
    labels = []

    def write(name, audio, label, onsets):
        save_wav(os.path.join(TEST_DIR, name), np.clip(audio, -1, 1))
        labels.append({"file": name, "label": label, "onsets": onsets})

    for cat, (fname, fn) in ALARMS.items():
        base = fname.replace(".wav", "")
        ref = reference_audio(cat)

        def variant(sec=6.0):
            # different start point in the cycle + slight pitch difference
            off = rng.uniform(0, 3)
            jit = rng.uniform(-0.015, 0.015)
            if ALARMS[cat][1] and not _custom_reference(cat):
                return fn(sec, jitter=jit, offset=off)
            need = int(sec * SR)
            src = np.tile(ref, int(np.ceil(2 * need / len(ref))))
            start = rng.integers(0, len(src) - need)
            return src[start:start + need]

        a, on = place(variant())
        write(f"{base}_original.wav", a, cat, [on])

        a, on = place(variant() * 0.08, amb_level=0.004)
        write(f"{base}_low_volume.wav", a, cat, [on])

        sig = variant()
        a, on = place(at_snr(sig, pink_noise(len(sig) / SR), 5))
        write(f"{base}_noise_5db.wav", a, cat, [on])

        sig = variant()
        a, on = place(at_snr(sig, speech_like(len(sig) / SR), 3),
                      ambience=speech_like, amb_level=0.3)
        write(f"{base}_with_speech.wav", a, cat, [on])

        a, on = place(far_away(variant()))
        write(f"{base}_distance.wav", a, cat, [on])

        # Re-arm test: alarm, 4 s silence, alarm again -> expect 2 detections
        s1, s2 = variant(5.0), variant(5.0)
        gap = np.zeros(int(4.0 * SR))
        a, on = place(np.concatenate([s1, gap, s2]))
        write(f"{base}_twice.wav", a, cat, [on, on + 9.0])

    write("neg_speech.wav", speech_like(10), "NONE", [])
    write("neg_music.wav", music_like(10), "NONE", [])
    write("neg_noise.wav", pink_noise(10) * 0.3, "NONE", [])
    write("neg_silence.wav", pink_noise(10) * 0.001, "NONE", [])
    write("neg_foreign_beep.wav", foreign_beep(10) * 0.6, "NONE", [])
    write("neg_speech_music_mix.wav",
          0.6 * speech_like(10) + 0.4 * music_like(10), "NONE", [])

    with open(os.path.join(TEST_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"{len(labels)} test files written to {TEST_DIR}")


_CUSTOM = set()


def _custom_reference(cat):
    return cat in _CUSTOM


if __name__ == "__main__":
    if "--tests-only" in sys.argv:
        # references are user recordings: build test variants from them
        _CUSTOM.update(ALARMS.keys())
    else:
        make_references()
    make_tests()
