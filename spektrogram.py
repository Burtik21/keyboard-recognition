import pyaudio
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import noisereduce as nr
from scipy.io.wavfile import write, read

# Parametry nahrávání
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3

# Inicializace nahrávání
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("Začínám nahrávat...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Nahrávání dokončeno.")

stream.stop_stream()
stream.close()
p.terminate()

# Konverze na numpy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
write("nahrany_zvuk.wav", RATE, audio_data)

# Načtení zvuku
rate, data = read("nahrany_zvuk.wav")

# High-pass filtr (odstranění nízkofrekvenčního šumu pod 300 Hz)
def highpass_filter(data, sr, cutoff=300):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(1, normal_cutoff, btype='high')
    return signal.filtfilt(b, a, data)

data_filtered = highpass_filter(data.astype(np.float32), RATE, cutoff=300)

# Jemnější spektrální odčítání šumu
data_denoised = nr.reduce_noise(y=data_filtered, sr=RATE, y_noise=data_filtered, prop_decrease=0.15)

# Band-pass filtr (ponechá pouze frekvence 300–8000 Hz)
def bandpass_filter(data, sr, lowcut=300, highcut=8000):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

data_final = bandpass_filter(data_denoised, RATE, lowcut=300, highcut=8000)

# Uložení finálního zvuku
write("cisty_zvuk.wav", RATE, (data_final * 32767).astype(np.int16))

# Spektrogram vyčištěného zvuku
D = librosa.amplitude_to_db(np.abs(librosa.stft(data_final)), ref=np.max)

# Vykreslení spektrogramu
plt.figure(figsize=(12, 6))
librosa.display.specshow(D, sr=RATE, x_axis='time', y_axis='log')

for t in np.arange(0, RECORD_SECONDS, 0.05):
    plt.axvline(x=t, color='white', linestyle='--', alpha=0.4)
for t in np.arange(0, RECORD_SECONDS, 0.2):
    plt.axvline(x=t, color='red', linestyle='-', alpha=0.7)

plt.colorbar(format='%+2.0f dB')
plt.title('Spektrogram vyčištěného zvuku (s redukcí šumu a band-pass filtrem)')
plt.xlabel('Čas (s)')
plt.ylabel('Frekvence (Hz)')
plt.show()
