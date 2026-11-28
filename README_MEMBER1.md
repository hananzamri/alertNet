# AlertNet – Member 1: Audio Detection and Signal Processing

## Setup
```bash
pip install -r requirements.txt
python tools/generate_sounds.py      # creates reference + test sounds
```
On Linux the microphone also needs PortAudio: `sudo apt install libportaudio2`.

## Files
| File | Purpose |
|---|---|
| `detection/audio_capture.py` | WAV loading, microphone stream (`MicrophoneStream`), WAV streaming for tests |
| `detection/audio_preprocessing.py` | stereo→mono, resampling to 22,050 Hz, normalization, band-pass noise filter (250–6000 Hz), windowing, RMS |
| `detection/feature_extraction.py` | FFT, spectrogram, 48 log-spaced frequency bands, cross-correlation, fingerprint + similarity |
| `detection/templates.py` | Loads the 3 reference WAVs and builds fingerprints |
| `detection/alarm_detector.py` | `AlarmDetector` – classification + trigger/re-arm logic |
| `client/monitor.py` | Computer A: microphone/WAV → detector → `HTTPClient.send_alert()` |
| `tools/generate_sounds.py` | Generates reference alarms and 24 test recordings |
| `tools/inspect_audio.py` | First task: sample rate, duration, waveform, FFT, spectrogram plots |
| `tests/test_detection.py` | Evaluation: accuracy, precision, recall, false positives, latency |

## How detection works
1. Audio is cut into **1 s windows** with **0.25 s hop**.
2. **RMS gate**: windows quieter than −50 dBFS are ignored (silence).
3. **Band-pass filter** 250–6000 Hz removes rumble and hiss.
4. **Spectrogram → 48 log-spaced bands.** Two fingerprints are computed:
   - *spectral profile* – which frequencies are present,
   - *temporal variation* – how they change (beeping, sweeping, two-tone).
5. **Tonality check**: alarms are tonal; flat spectra (noise, speech) are rejected.
6. **Similarity** (correlation) with every window of each reference sound.
   Best score = confidence. Needs ≥ 0.85 and a clear margin over the 2nd category.
7. **Confirmation**: 3 consecutive matching windows → one detection.
8. **No repeated alerts**: after a detection the detector waits until the
   alarm has been absent for 2.5 s before it can trigger again.

Why correlation of compressed power instead of dB: background noise adds a
nearly constant offset across bands, which correlation ignores. This made
detection in 5 dB SNR noise and speech work (recall went from 62 % to 100 %).

## Interface (used by Member 2)
```python
from detection.templates import load_templates
from detection.alarm_detector import AlarmDetector

detector = AlarmDetector(load_templates(), threshold=0.85)
result = detector.process_audio(chunk, 22050)
# None  or  {"alarm_type": "FIRE_ALARM", "confidence": 0.95}
```
`process_audio` accepts any chunk length (mic blocks or a whole file).

## Run
```bash
python tools/inspect_audio.py                       # plots -> sounds/plots/
python tests/test_detection.py                      # evaluation report
python -m pytest tests/test_detection.py            # pass/fail tests
python -m client.monitor --dry-run --wav sounds/test/fire_alarm_original.wav
python -m client.monitor --server 192.168.1.10      # live mic → server
python -m client.monitor --server 192.168.1.10 --wav sounds/reference/fire_alarm.wav --realtime
```
`--verbose` prints the score of every window (useful for tuning threshold).

## Test results (sounds/test, 24 files)
| Metric | Result |
|---|---|
| Accuracy (files fully correct) | 100 % |
| Precision | 100 % |
| Recall | 100 % |
| False positives (speech, music, noise, silence, foreign beep) | 0 |
| Detection latency | mean 1.09 s, max 2.0 s |

Conditions per alarm: original, low volume (−22 dB), pink noise at 5 dB SNR,
mixed with speech, distance (low-pass + echo), played twice (re-arm test).

Extra stress test on unseen random variants: all categories 5/5 at 10 and
5 dB SNR; at 0 dB SNR fire and general warning 5/5, security siren 2/5.
**Limitation:** the sweeping siren is the weakest category in very loud noise.

## Using real recordings
The reference sounds are synthesized. To use real alarm recordings, put them in
`sounds/reference/` with the same names (≥ 3 s, any sample rate, mono or
stereo) and run:
```bash
python tools/generate_sounds.py --tests-only
python tests/test_detection.py
```
Then re-check the threshold with `python tests/test_detection.py 0.8` etc.
