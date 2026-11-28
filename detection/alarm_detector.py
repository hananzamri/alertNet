"""
Alarm detector for AlertNet (Member 1).

Pipeline per 1-second window (hop 0.25 s):
  preprocess -> RMS gate -> features -> compare with templates -> score

Decision logic (state machine):
  ARMED      : the same category must match in `consecutive` windows in a
               row (default 3) -> one detection is returned, go to TRIGGERED.
  TRIGGERED  : no new detection while the alarm keeps playing. When no
               alarm has matched for `rearm_seconds` (default 2.5 s),
               go back to ARMED.

Usage:
    templates = load_templates()
    detector = AlarmDetector(templates)
    for chunk in mic.chunks():
        result = detector.process_audio(chunk, 22050)
        if result:
            http_client.send_alert("DEVICE_A", result["alarm_type"],
                                   result["confidence"])
"""

import numpy as np

from detection.audio_capture import TARGET_SR
from detection.audio_preprocessing import preprocess, resample, rms_db
from detection.feature_extraction import extract_features, feature_similarity


class AlarmDetector:
    def __init__(self, templates, threshold=0.85,
                 sample_rate=TARGET_SR, window_seconds=1.0, hop_seconds=0.25,
                 consecutive=3, rearm_seconds=2.5,
                 min_rms_db=-50.0, min_tonality_db=6.0, min_margin=0.04):
        self.templates = templates
        self.threshold = threshold
        self.sr = sample_rate
        self.win = int(window_seconds * sample_rate)
        self.hop = int(hop_seconds * sample_rate)
        self.hop_seconds = hop_seconds
        self.consecutive = consecutive
        self.rearm_windows = int(np.ceil(rearm_seconds / hop_seconds))
        self.min_rms_db = min_rms_db
        self.min_tonality_db = min_tonality_db
        self.min_margin = min_margin
        self.reset()

    # ------------------------------------------------------------ state --
    def reset(self):
        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_seen = 0          # samples already consumed (stream clock)
        self.armed = True
        self._candidate = None
        self._run = []                  # scores of current matching run
        self._absent = 0
        self.last_window = None         # debug info about the latest window
        self.last_detection_time = None # stream time (s) of the last detection

    # --------------------------------------------------- classification --
    def classify_window(self, window):
        """
        Classify one window (already at self.sr).
        Returns (category or None, score, scores_per_category).
        """
        level = rms_db(window)
        if level < self.min_rms_db:
            return None, 0.0, {"reason": "silence", "rms_db": level}

        audio = preprocess(window, self.sr)
        feats = extract_features(audio, self.sr)
        if feats["tonality"] < self.min_tonality_db:
            return None, 0.0, {"reason": "not tonal",
                               "tonality": feats["tonality"]}

        scores = {}
        for t in self.templates:
            scores[t.category] = max(feature_similarity(feats, ref)
                                     for ref in t.features)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_cat, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best >= self.threshold and best - second >= self.min_margin:
            return best_cat, best, scores
        return None, best, scores

    # ------------------------------------------------------- streaming --
    def process_audio(self, audio, sample_rate):
        """
        Feed any amount of audio (a mic chunk or a whole file).

        Returns None if no alarm is detected.
        Otherwise returns:
            {"alarm_type": "FIRE_ALARM", "confidence": 0.95}
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != self.sr:
            audio = resample(audio, sample_rate, self.sr)

        self._buffer = np.concatenate([self._buffer, audio])
        result = None

        while len(self._buffer) >= self.win:
            window = self._buffer[:self.win]
            detection = self._step(window)
            self._buffer = self._buffer[self.hop:]
            self._samples_seen += self.hop
            if detection and result is None:
                result = detection
        return result

    def _step(self, window):
        category, score, scores = self.classify_window(window)
        self.last_window = {"category": category, "score": score,
                            "scores": scores}

        if not self.armed:
            self._absent = 0 if category else self._absent + 1
            if self._absent >= self.rearm_windows:
                self.armed = True
                self._candidate, self._run = None, []
            return None

        if category is None:
            self._candidate, self._run = None, []
            return None
        if category != self._candidate:
            self._candidate, self._run = category, []
        self._run.append(score)

        if len(self._run) >= self.consecutive:
            confidence = round(float(np.mean(self._run[-self.consecutive:])), 2)
            self.armed = False
            self._absent = 0
            # end of current window on the stream clock
            self.last_detection_time = (self._samples_seen + self.win) / self.sr
            self._candidate, self._run = None, []
            return {"alarm_type": category, "confidence": confidence}
        return None


def detect_file(path, detector, chunk_seconds=0.25):
    """
    Run a WAV file through the detector as if it were live audio.
    Returns list of (time_seconds, result) for every detection.
    """
    from detection.audio_capture import wav_stream
    detector.reset()
    found = []
    for chunk in wav_stream(path, chunk_seconds, target_sr=detector.sr):
        r = detector.process_audio(chunk, detector.sr)
        if r:
            found.append((detector.last_detection_time, r))
    return found
