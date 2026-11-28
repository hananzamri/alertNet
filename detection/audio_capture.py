"""
Audio capture for AlertNet (Member 1).

Two input sources share one interface (chunks of mono float32 audio):
  * WAV files      -> load_wav(), wav_stream()
  * Microphone     -> MicrophoneStream (needs the `sounddevice` package)

All audio is converted to mono float32 in [-1, 1] at TARGET_SR (22,050 Hz).
"""

import queue
import time

import numpy as np
from scipy.io import wavfile

from detection.audio_preprocessing import to_mono, resample, pcm_to_float

TARGET_SR = 22050


def load_wav(path, target_sr=TARGET_SR):
    """Load a WAV file as mono float32 at target_sr. Returns (audio, sr)."""
    sr, data = wavfile.read(path)
    audio = pcm_to_float(data)
    audio = to_mono(audio)
    audio = resample(audio, sr, target_sr)
    return audio, target_sr


def save_wav(path, audio, sr=TARGET_SR):
    """Save float audio in [-1, 1] as 16-bit PCM WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))


def wav_stream(path, chunk_seconds=0.25, realtime=False, target_sr=TARGET_SR):
    """
    Yield a WAV file chunk by chunk, imitating a live microphone.
    realtime=True sleeps between chunks (useful for demo / latency tests).
    """
    audio, sr = load_wav(path, target_sr)
    chunk = int(chunk_seconds * sr)
    for start in range(0, len(audio), chunk):
        yield audio[start:start + chunk]
        if realtime:
            time.sleep(chunk_seconds)


class MicrophoneStream:
    """
    Continuous microphone capture.

        with MicrophoneStream() as mic:
            for chunk in mic.chunks():
                detector.process_audio(chunk, mic.sample_rate)
    """

    def __init__(self, sample_rate=TARGET_SR, chunk_seconds=0.25, device=None):
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.device = device
        self._queue = queue.Queue()
        self._stream = None
        self.running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        self._queue.put(indata.copy())

    def start(self):
        import sounddevice as sd  # imported here so WAV mode works without it
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self.running = True

    def stop(self):
        self.running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def chunks(self, timeout=1.0):
        """Generator of mono float32 chunks while the stream is running."""
        while self.running:
            try:
                block = self._queue.get(timeout=timeout)
            except queue.Empty:
                continue
            yield to_mono(block)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def list_input_devices():
    """Print available microphones (helps pick the right device)."""
    import sounddevice as sd
    print(sd.query_devices())
