import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import soundfile as sf

class ImpulseRecorder:
    def __init__(self, rate=44100, channels=1, chunk=1024, record_seconds=1, threshold_ratio=0.3):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.threshold_ratio = threshold_ratio
        self.format = pyaudio.paInt16

    def record_audio(self, filename="impulz.wav"):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True,
                        frames_per_buffer=self.chunk)

        print("🎙️ Nahrávám zvuk...")
        frames = []
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("✅ Nahráno!")

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        write(filename, self.rate, audio_data)
        return filename

    def measure_impulse_duration(self, filename):
        y, sr = sf.read(filename)
        if len(y.shape) > 1:  # stereo na mono
            y = np.mean(y, axis=1)

        # RMS energie
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]

        # Graf RMS energie
        plt.figure(figsize=(10, 4))
        plt.plot(librosa.times_like(rms, sr=sr, hop_length=512), rms)
        plt.xlabel('Čas (s)')
        plt.ylabel('RMS energie')
        plt.title('RMS energie vzorku')
        plt.grid(True)
        plt.show()

        threshold = self.threshold_ratio * np.max(rms)
        above_thresh = np.where(rms > threshold)[0]

        if len(above_thresh) == 0:
            print("⚠️ Žádný impulz nenalezen!")
            return None

        start_time = librosa.frames_to_time(above_thresh[0], sr=sr, hop_length=512)
        end_time = librosa.frames_to_time(above_thresh[-1], sr=sr, hop_length=512)
        duration = end_time - start_time

        print(f"🕒 Délka impulzu: {duration:.4f} sekund")
        return duration
# Použití třídy
recorder = ImpulseRecorder(record_seconds=1)
filename = recorder.record_audio()
impulse_duration = recorder.measure_impulse_duration(filename)
