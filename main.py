from app.audio_recorder import AudioRecorder
from app.audio_sender import AudioSender
from app.event_detector import AudioDetector
import time
import config
import numpy as np

recorder = AudioRecorder()
detector = AudioDetector(rate=recorder.rate, threshold=config.RMS_THRESHOLD)
sender = AudioSender()

last_detection_time = 0

print("🎙️ Spouštím real-time detekci...")

try:
    while True:
        audio = recorder.record_audio(duration=config.WINDOW_DURATION)
        print(f"📊 Audio info → dtype: {audio.dtype}, min: {np.min(audio):.4f}, max: {np.max(audio):.4f}")

        result = detector.detect_rms_impulse(audio)
        current_time = time.time()

        if result and current_time - last_detection_time > config.COOLDOWN_SECONDS:
            print("✅ Zvuk detekován, posílám...")

            # Volitelně zobrazit RMS hodnotu, která spustila detekci
            rms = detector.compute_rms_framewise(audio)
            trigger_value = np.max(rms)
            print(f"🎯 RMS hodnota, která spustila detekci: {trigger_value:.6f}")

            sender.send_wav(audio)
            last_detection_time = current_time
        else:
            print("...")

except KeyboardInterrupt:
    print("\n🛑 Ukončeno uživatelem.")
