from audio_utils import AudioRecorder, AudioProcessor
import pandas as pd
import time
import os

recorder = AudioRecorder()
processor = AudioProcessor()
csv_filename = 'data.csv'

data = []

for i in range(50):
    print(f"\n🎙️ [{i + 1}/50] Připrav se na záznam. Teď!")

    audio = recorder.record_audio(duration=1)
    filtered_audio = processor.apply_filters(audio)

    # Detekuj impulzy (vrací seznam)
    impulses = processor.detect_impulses(filtered_audio, threshold=0.0011j)


    if impulses:
        start, end, duration = impulses[0]  # vezmeme jen první
        print(f"✅ Impulz: začátek={start:.4f}s, konec={end:.4f}s, trvání={duration:.4f}s")

        impulse_audio = filtered_audio[int(start * recorder.rate):int(end * recorder.rate)]
        features = processor.extract_features(impulse_audio)
        features['duration'] = duration
        features['sound_category'] = '1'  # nebo 'klavesnice', jak chceš

        data.append(features)
        print("✅ Features uloženy.")
    else:
        print("⚠️ Žádný impulz nenalezen. Vzorek přeskočen.")

    print("⏳ Konec záznamu. Čekám 1 sekundu.")
    time.sleep(0.5)

# Uložení do CSV
df = pd.DataFrame(data)
df.to_csv(csv_filename, index=False, mode='a', header=not os.path.exists(csv_filename))

print(f"\n🎉 Hotovo! Data uložena do '{csv_filename}'.")
