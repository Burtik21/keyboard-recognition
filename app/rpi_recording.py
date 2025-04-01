from audio_utils import AudioRecorder
from pyAudioAnalysis import ShortTermFeatures
import numpy as np
import time

recorder = AudioRecorder()

THRESHOLD = 0.0011
WINDOW_DURATION = 0.5  # sekundy
COOLDOWN_SECONDS = 0.2

last_detection_time = 0

print("🎙️ Spouštím real-time detekci... (Ctrl+C pro ukončení)")

try:
    while True:
        print("✅ Skript spuštěn!")

        audio = recorder.record_audio(duration=WINDOW_DURATION)
        audio = audio.flatten()

        # Výpočet RMS a ZCR pomocí pyAudioAnalysis
        win_size = int(0.025 * recorder.rate)  # 25 ms
        step_size = int(0.01 * recorder.rate)  # 10 ms
        features, _ = ShortTermFeatures.feature_extraction(audio, recorder.rate, win_size, step_size)

        rms_energy = features[1]  # Short-term energy ≈ RMS²
        zcr = features[0]

        max_rms = np.max(rms_energy)
        current_time = time.time()

        if max_rms > THRESHOLD and current_time - last_detection_time > COOLDOWN_SECONDS:
            print("✅ Zvuk detekován!")
            duration = WINDOW_DURATION  # celé okno (přesněji bychom museli dopočítat)
            print(f"📈 Zvuk trvá {duration:.4f}s | Max RMS: {max_rms:.6f} | Avg ZCR: {np.mean(zcr):.4f}")
            last_detection_time = current_time
        else:
            print("...")

except KeyboardInterrupt:
    print("\n🛑 Detekce ukončena uživatelem.")
