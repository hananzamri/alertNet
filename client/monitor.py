"""
Monitoring client – Computer A (Member 1 + Member 2 integration).

Listens to the microphone (or plays a WAV file through the detector),
and when an alarm is detected sends it to the server with Member 2's
HTTPClient (POST /alerts).

    # live microphone
    python -m client.monitor --server 192.168.1.10 --device DEVICE_A

    # test mode with a recording
    python -m client.monitor --server 192.168.1.10 --wav sounds/test/fire_alarm_original.wav

    # no server yet (detection only)
    python -m client.monitor --dry-run --wav sounds/test/fire_alarm_original.wav
"""

import argparse
import time

from detection.alarm_detector import AlarmDetector
from detection.audio_capture import MicrophoneStream, wav_stream, TARGET_SR
from detection.templates import load_templates


class DryRunClient:
    """Stand-in for HTTPClient while the server is not ready."""

    def register(self, device_id, role):
        print(f"[dry-run] register {device_id} as {role}")

    def send_alert(self, device_id, alarm_type, confidence):
        print(f"[dry-run] POST /alerts {device_id} {alarm_type} {confidence}")


def make_client(args):
    if args.dry_run:
        return DryRunClient()
    from client.http_client import HTTPClient  # Member 2's module
    return HTTPClient(args.server, args.port)


def on_detection(client, device_id, result):
    print(f"[{time.strftime('%H:%M:%S')}] DETECTED {result['alarm_type']} "
          f"(confidence {result['confidence']})")
    try:
        response = client.send_alert(device_id, result["alarm_type"],
                                     result["confidence"])
        if response is not None:
            print(f"  server: {response}")
    except Exception as exc:  # network problems must not stop monitoring
        print(f"  could not send alert: {exc}")


def main():
    ap = argparse.ArgumentParser(description="AlertNet monitoring client")
    ap.add_argument("--server", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="DEVICE_A")
    ap.add_argument("--wav", help="test mode: analyse this WAV file")
    ap.add_argument("--realtime", action="store_true",
                    help="play WAV at real speed (demo / latency)")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--mic-device", default=None, help="microphone index/name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="print the score of every window")
    args = ap.parse_args()

    detector = AlarmDetector(load_templates(), threshold=args.threshold)
    client = make_client(args)
    try:
        client.register(args.device, "MONITOR")
    except Exception as exc:
        print(f"registration failed: {exc}")

    if args.wav:
        source = wav_stream(args.wav, 0.25, realtime=args.realtime)
        print(f"Analysing {args.wav} ...")
    else:
        mic = MicrophoneStream(TARGET_SR, 0.25, device=args.mic_device)
        mic.start()
        source = mic.chunks()
        print("Listening to microphone... (Ctrl+C to stop)")

    try:
        for chunk in source:
            result = detector.process_audio(chunk, TARGET_SR)
            if args.verbose and detector.last_window:
                w = detector.last_window
                print(f"  window: {w['category']} {w['score']:.2f}")
            if result:
                on_detection(client, args.device, result)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.wav:
            mic.stop()
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()
