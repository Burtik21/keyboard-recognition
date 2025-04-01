import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_utils import AudioProcessor  # 👈 Uprav podle cesty ke třídě

filename = "nahravka_6.webm"  # nebo třeba sys.argv[1] podle potřeby

# 1. Načti WAV
y, sr = sf.read(filename)
if y.ndim > 1:
    y = y.mean(axis=1)  # stereo -> mono

# 2. Aplikuj filtry
processor = AudioProcessor(rate=sr)
filtered_audio = processor.apply_filters(y)

# 3. RMS
rms = librosa.feature.rms(y=filtered_audio, frame_length=1024, hop_length=256)[0]
times_rms = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=256)

plt.figure(figsize=(12, 4))
plt.plot(times_rms, rms, label='RMS', color='orange')
plt.axhline(0.0015, color='red', linestyle='--', label='Threshold')
plt.xlabel('Čas (s)')
plt.ylabel('RMS')
plt.title('RMS energie po aplikaci filtrů')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 4. Spektrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio)), ref=np.max)

plt.figure(figsize=(12, 5))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spektrogram (filtrovaný)')
plt.tight_layout()
plt.show()
