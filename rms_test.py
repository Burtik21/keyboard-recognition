import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
THRESHOLD = 0.0015  # Tvůj práh

def record_audio(duration=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎙️ Nahrávám 1 sekundu...")
    frames = []
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("✅ Nahrávání dokončeno.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data

def analyze_audio(audio, sr, threshold):
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=256)

    # Graf RMS energie pro kontrolu
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms, label='RMS energie', color='orange')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Práh RMS (0.0015)')
    plt.xlabel('Čas (s)')
    plt.ylabel('RMS')
    plt.title('RMS energie s prahem')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Vykreslení spektrogramu
    plt.figure(figsize=(12, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spektrogram nahrávky')
    plt.xlabel('Čas (s)')
    plt.ylabel('Frekvence (Hz)')
    plt.show()

if __name__ == "__main__":
    audio = record_audio(duration=2)
    analyze_audio(audio, sr=RATE, threshold=THRESHOLD)
