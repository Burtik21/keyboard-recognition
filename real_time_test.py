import time
import pickle
import pandas as pd
from audio_utils import AudioRecorder, AudioProcessor

# === Config ===
THRESHOLD = 0.0010
DURATION = 0.5       # délka jednoho okna
SLEEP_TIME = 0.25    # překrývání
COOLDOWN = 0.0     # mezi kliky

input_features = [
    'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth',
    'spectral_rolloff', 'max_intensity', 'dominant_freq', 'duration',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_6',
    'mfcc_7', 'mfcc_8', 'mfcc_13'
]

label_map = {
    0: 'tleskani',
    1: 'klavesnice',
    2: 'bouchani'
}

# === Load model ===
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# === Init ===
recorder = AudioRecorder()
processor = AudioProcessor()
last_click_time = 0
keyboard_clicks = 0

print("🎧 Realtime detekce spuštěna – 0.5s okna, overlap 0.25s.\n")

try:
    while True:
        audio = recorder.record_audio(duration=DURATION)
        filtered_audio = processor.apply_filters(audio)

        # 👇 nový detect_impulses() musíš mít přidaný v AudioProcessor
        impulses = processor.detect_impulses(filtered_audio, threshold=THRESHOLD)

        for start, end, dur in impulses:
            now = time.time()
            if now - last_click_time >= COOLDOWN:
                impulse_audio = filtered_audio[int(start * recorder.rate):int(end * recorder.rate)]
                features = processor.extract_features(impulse_audio)
                features['duration'] = dur

                x_input = pd.DataFrame([features])[input_features]
                pred = int(model.predict(x_input)[0])
                label = label_map.get(pred, f'neznámé ({pred})')

                if pred == 1:
                    keyboard_clicks += 1
                    print(f"⌨️  Klávesa! (#{keyboard_clicks})")
                else:
                    print(f"🔉 Detekováno: {label} (trvání: {dur:.3f}s)")

                last_click_time = now

        time.sleep(SLEEP_TIME)

except KeyboardInterrupt:
    print(f"\n🛑 Ukončeno. Celkem detekovaných kláves: {keyboard_clicks}")
