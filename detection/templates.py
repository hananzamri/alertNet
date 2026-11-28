"""
Reference alarm templates for AlertNet (Member 1).

Each reference WAV (sounds/reference/*.wav) is cut into 1-second windows
and a fingerprint is stored for every window. A captured window is compared
with all stored fingerprints of a category; the best match is its score.
Storing several windows makes matching independent of WHEN in the alarm
cycle the microphone started listening.
"""

import os

from detection.audio_capture import load_wav, TARGET_SR
from detection.audio_preprocessing import preprocess, frame_signal, rms_db
from detection.feature_extraction import extract_features

# file name -> category sent to the server
CATEGORY_FILES = {
    "FIRE_ALARM": "fire_alarm.wav",
    "SECURITY_ALARM": "security_alarm.wav",
    "GENERAL_WARNING": "general_warning.wav",
}

DEFAULT_REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sounds", "reference",
)


class AlarmTemplate:
    def __init__(self, category, features):
        self.category = category
        self.features = features  # list of feature dicts

    def __repr__(self):
        return f"AlarmTemplate({self.category}, {len(self.features)} windows)"


def build_template(category, audio, sr, window_seconds=1.0, hop_seconds=0.25):
    audio = preprocess(audio, sr)
    windows = frame_signal(audio, sr, window_seconds, hop_seconds)
    feats = [extract_features(w, sr) for w in windows if rms_db(w) > -60]
    if not feats:
        raise ValueError(f"Reference for {category} is silent or too short "
                         f"(needs at least {window_seconds} s of sound)")
    return AlarmTemplate(category, feats)


def load_templates(reference_dir=DEFAULT_REFERENCE_DIR, sr=TARGET_SR,
                   window_seconds=1.0, hop_seconds=0.25):
    """Load the three reference alarms. Returns list[AlarmTemplate]."""
    templates = []
    for category, filename in CATEGORY_FILES.items():
        path = os.path.join(reference_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing reference sound: {path}")
        audio, _ = load_wav(path, sr)
        templates.append(build_template(category, audio, sr,
                                        window_seconds, hop_seconds))
    return templates
