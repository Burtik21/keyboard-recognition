from audio_utils import AudioRecorder, AudioProcessor
import pandas as pd
import time
import os

recorder = AudioRecorder()
processor = AudioProcessor()
csv_filename = 'keyboard_data.csv'

# Připrav seznam pro data
data = []

for i in range(50):
    print(f"\n🎙️ [{i + 1}/100] Připrav se na záznam. Teď!")

    audio = recorder.record_audio(duration=1)

    # Aplikuj filtry
    filtered_audio = processor.apply_filters(audio)

    # Detekuj impulz
    start, end, duration = processor.detect_impulse(filtered_audio, threshold=0.0015)

    if duration is None:
        print("⚠️ Žádný impulz nenalezen. Vzorek přeskočen.")
    else:
        print(f"✅ Impulz: začátek={start:.4f}s, konec={end:.4f}s, trvání={duration:.4f}s")

        # Extrahuj features
        impulse_audio = filtered_audio[int(start * recorder.rate):int(end * recorder.rate)]
        features = processor.extract_features(impulse_audio)

        # Přidej další informace (duration, sound_category)
        features['duration'] = duration
        features['sound_category'] = 'vykrik'

        # Přidej do seznamu dat
        data.append(features)

        print("✅ Features uloženy.")

    print("Konec záznamu. Čekám 2 sekundy.")
    time.sleep(1)

# Uložení dat do CSV souboru
df = pd.DataFrame(data)
df.to_csv(csv_filename, index=False, mode='a', header=not os.path.exists(csv_filename))

print(f"\n🎉 Hotovo! Data uložena do '{csv_filename}'.")
