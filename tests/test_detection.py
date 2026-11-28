"""
Detection tests and evaluation for AlertNet (Member 1).

    python tests/test_detection.py            # full evaluation report
    python -m pytest tests/test_detection.py  # pass/fail checks

Measures accuracy, precision, recall, false positives and detection latency
on sounds/test/ (labels in sounds/test/labels.json).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection.templates import load_templates  # noqa: E402
from detection.alarm_detector import AlarmDetector, detect_file  # noqa: E402

TEST_DIR = os.path.join(ROOT, "sounds", "test")
CATEGORIES = ["FIRE_ALARM", "SECURITY_ALARM", "GENERAL_WARNING"]


def evaluate(threshold=0.85, verbose=True):
    templates = load_templates()
    detector = AlarmDetector(templates, threshold=threshold)
    with open(os.path.join(TEST_DIR, "labels.json")) as f:
        labels = json.load(f)

    rows, latencies = [], []
    tp = {c: 0 for c in CATEGORIES}
    fp = {c: 0 for c in CATEGORIES}
    fn = {c: 0 for c in CATEGORIES}
    correct_files = 0

    for item in labels:
        found = detect_file(os.path.join(TEST_DIR, item["file"]), detector)
        expected = item["label"]
        onsets = item["onsets"]
        predicted = [r["alarm_type"] for _, r in found]

        # match each expected onset with the first correct detection after it
        used = set()
        for on in onsets:
            hit = None
            for i, (t, r) in enumerate(found):
                if i not in used and r["alarm_type"] == expected and t >= on:
                    hit = i
                    break
            if hit is None:
                fn[expected] += 1
            else:
                used.add(hit)
                tp[expected] += 1
                latencies.append(found[hit][0] - on)
        for i, (_, r) in enumerate(found):
            if i not in used:
                fp[r["alarm_type"]] += 1

        ok = (len(found) == len(onsets) and len(used) == len(onsets))
        correct_files += ok
        rows.append((item["file"], expected, predicted,
                     [r["confidence"] for _, r in found], ok))

    total_tp, total_fp, total_fn = (sum(d.values()) for d in (tp, fp, fn))
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
    accuracy = correct_files / len(labels)
    report = {
        "threshold": threshold,
        "files": len(labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positives": total_fp,
        "missed": total_fn,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "per_category": {c: {"tp": tp[c], "fp": fp[c], "fn": fn[c]}
                         for c in CATEGORIES},
        "rows": rows,
    }

    if verbose:
        print(f"\n{'FILE':34} {'EXPECTED':16} {'DETECTED':32} RESULT")
        print("-" * 92)
        for name, exp, pred, conf, ok in rows:
            det = ", ".join(f"{p}({c})" for p, c in zip(pred, conf)) or "-"
            print(f"{name:34} {exp:16} {det:32} {'OK' if ok else 'FAIL'}")
        print("-" * 92)
        print(f"Accuracy (files fully correct): {accuracy:.1%}")
        print(f"Precision: {precision:.1%}   Recall: {recall:.1%}")
        print(f"False positives: {total_fp}   Missed alarms: {total_fn}")
        if latencies:
            print(f"Detection latency: mean {report['mean_latency_s']:.2f} s, "
                  f"max {report['max_latency_s']:.2f} s")
        for c in CATEGORIES:
            print(f"  {c:16} TP={tp[c]} FP={fp[c]} FN={fn[c]}")
    return report


# ------------------------------------------------------- pytest checks --
def test_reference_files_detected():
    templates = load_templates()
    det = AlarmDetector(templates)
    for cat, fname in [("FIRE_ALARM", "fire_alarm_original.wav"),
                       ("SECURITY_ALARM", "security_alarm_original.wav"),
                       ("GENERAL_WARNING", "general_warning_original.wav")]:
        found = detect_file(os.path.join(TEST_DIR, fname), det)
        assert len(found) == 1 and found[0][1]["alarm_type"] == cat


def test_no_false_alarm_on_speech_and_music():
    templates = load_templates()
    det = AlarmDetector(templates)
    for fname in ["neg_speech.wav", "neg_music.wav", "neg_noise.wav"]:
        assert detect_file(os.path.join(TEST_DIR, fname), det) == []


def test_single_trigger_and_rearm():
    templates = load_templates()
    det = AlarmDetector(templates)
    found = detect_file(os.path.join(TEST_DIR, "fire_alarm_twice.wav"), det)
    assert len(found) == 2


def test_overall_quality():
    r = evaluate(verbose=False)
    assert r["precision"] >= 0.9 and r["recall"] >= 0.8


if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
    evaluate(th)
