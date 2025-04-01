import sounddevice as sd
import numpy as np
import pandas as pd
import librosa
import joblib

# === 1. Načtení modelu ===
model = joblib.load("keyboard_model.pkl")

# === 2. Nahrávání nového zvuku ===
DURATION = 3  # 3 sekundy pro rychlý test
SAMPLE_RATE = 44100

print("Nahrávám nový zvuk...")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("Nahrávání dokončeno!")

# === 3. Předzpracování zvuku ===
y = np.array(audio_data.flatten(), dtype=float
# Funkce pro extrakci stejných vlastností jako při trénování
def extract_features(y, sr):
    return {
        "rms": np.sqrt(np.mean(y ** 2)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    }

# Extrahování vlastností
features = extract_features(y, SAMPLE_RATE)
X_new = pd.DataFrame([features])  # Převod na DataFrame

# === 4. Predikce ===
prediction = model.predict(X_new)

print("Výsledek modelu:", "Pressing DETEKOVÁN" if prediction[0] == 1 else "Pressing NENÍ")
