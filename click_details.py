from audio_utils import AudioRecorder, AudioProcessor
import numpy as np
import librosa
import time

recorder = AudioRecorder()
processor = AudioProcessor()

THRESHOLD = 0.0011
WINDOW_DURATION = 0.5  # sekundy
COOLDOWN_SECONDS = 0.2

last_detection_time = 0

print("🎙️ Spouštím real-time detekci... (Ctrl+C pro ukončení)")

try:
    while True:
        audio = recorder.record_audio(duration=WINDOW_DURATION)
        filtered_audio = processor.apply_filters(audio)

        # RMS & ZCR
        rms = librosa.feature.rms(y=filtered_audio, frame_length=1024, hop_length=256)[0]
        zcr_full = librosa.feature.zero_crossing_rate(y=filtered_audio, frame_length=1024, hop_length=256)[0]

        current_time = time.time()

        if np.max(rms) > THRESHOLD and current_time - last_detection_time > COOLDOWN_SECONDS:
            print("✅ Zvuk detekován!")

            above_threshold = rms > THRESHOLD
            impulse_start = np.argmax(above_threshold)
            impulse_end = len(rms) - np.argmax(above_threshold[::-1])

            start_time = librosa.frames_to_time(impulse_start, sr=recorder.rate, hop_length=256)
            end_time = librosa.frames_to_time(impulse_end, sr=recorder.rate, hop_length=256)
            duration = end_time - start_time

            impulse_audio = filtered_audio[int(start_time * recorder.rate):int(end_time * recorder.rate)]
            features = processor.extract_features(impulse_audio)
            features['duration'] = duration
            features['avg_zcr'] = float(np.mean(zcr_full[impulse_start:impulse_end]))

            print(f"📈 Zvuk trvá {duration:.4f}s | Features: {features}")
            last_detection_time = current_time
        else:
            print("...")

except KeyboardInterrupt:
    print("\n🛑 Detekce ukončena uživatelem.")
