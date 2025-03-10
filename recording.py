import os

import sounddevice as sd
import numpy as np
import pandas as pd
import librosa

DURATION = 30  # Doba nahrávání v sekundách
SAMPLE_RATE = 44100  # Vzorkovací frekvence
WINDOW_SIZE = 0.025  # Velikost okna pro analýzu v sekundách (25 ms)
HOP_SIZE = 0.01  # Posun mezi okny (10 ms)

# Funkce pro extrakci vlastností ze zvuku (v okně)
def extract_features(y, sr):
    features = {}

    # RMS energie
    rms = np.sqrt(np.mean(y ** 2))
    features['rms'] = rms

    # Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr'] = np.mean(zcr)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth)

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features['spectral_rolloff'] = np.mean(spectral_rolloff)

    return features


# === 1. Nahrávání zvuku ===
print("Začínám nahrávání...")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("Nahrávání dokončeno!")

# === 2. Načtení zvuku ===
y = np.array(audio_data.flatten(), dtype=float)  # Převedení na jedno-dimenzionální pole
sr = SAMPLE_RATE

# === 3. Segmentace a analýza ===
window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
hop_samples = int(HOP_SIZE * SAMPLE_RATE)
num_windows = (len(y) - window_samples) // hop_samples

# === 4. Ruční zadání stavu (pressing) ===
pressing = 0

# === 5. Extrakce vlastností pro každé okno ===
data = []
for i in range(num_windows):
    start = i * hop_samples
    end = start + window_samples
    window_data = y[start:end]

    # Extrahování vlastností
    features = extract_features(window_data, sr)
    features['pressing'] = pressing

    data.append(features)

# === 6. Uložení dat do CSV ===
df = pd.DataFrame(data)
df.to_csv('keyboard_final_data.csv', mode='a', index=False, header=not os.path.exists('keyboard_final_data.csv'))

print("Data byla úspěšně uložena do 'keyboard_recognition_data.csv'.")
