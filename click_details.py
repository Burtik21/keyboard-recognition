from audio_utils import AudioRecorder, AudioProcessor
import numpy as np
import librosa

recorder = AudioRecorder()
processor = AudioProcessor()

# Nahraj zvuk (1 sekunda)
audio = recorder.record_audio(duration=1)

# Aplikuj filtry
filtered_audio = processor.apply_filters(audio)

# Vykresli RMS a spektrogram
processor.plot_rms(filtered_audio, threshold=0.0015)
processor.plot_spectrogram(filtered_audio)

# Zjisti RMS
rms = librosa.feature.rms(y=filtered_audio, frame_length=1024, hop_length=256)[0]

# Detekce obecného zvuku (ne impulzu)
if np.max(rms) > 0.0015:
    print("✅ Zvuk detekován! (např. křik nebo cokoliv hlasitého)")

    # Najdi začátek a konec zvuku
    above_threshold = rms > 0.0015
    impulse_start = np.argmax(above_threshold)
    impulse_end = len(rms) - np.argmax(above_threshold[::-1])  # z druhé strany

    start_time = librosa.frames_to_time(impulse_start, sr=recorder.rate, hop_length=256)
    end_time = librosa.frames_to_time(impulse_end, sr=recorder.rate, hop_length=256)
    duration = end_time - start_time

    print(f"📈 Zvuk trvá přibližně {duration:.4f} s (od {start_time:.4f} do {end_time:.4f})")

    # Vyřízni část audia a spočítej features
    impulse_audio = filtered_audio[int(start_time*recorder.rate):int(end_time*recorder.rate)]
    features = processor.extract_features(impulse_audio)
    features['duration'] = duration
    print("✅ Features:", features)
else:
    print("⚠️ Zvuk nebyl detekován – ticho nebo příliš slabý.")
