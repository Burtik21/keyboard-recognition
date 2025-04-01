from app.audio_recorder import AudioRecorder
from app.audio_sender import AudioSender
from app.event_detector import AudioDetector

import time
import config

recorder = AudioRecorder()
detector = AudioDetector(rate=recorder.rate)

recorder = AudioRecorder()
detector = AudioDetector(threshold=config.RMS_THRESHOLD)
sender = AudioSender()

last_detection_time = 0

print("🎙️ Spouštím real-time detekci...")

try:
    while True:
        audio = recorder.record_audio(duration=config.WINDOW_DURATION)
        result = detector.detect_rms_impulse(audio)

        current_time = time.time()

        if result and current_time - last_detection_time > config.COOLDOWN_SECONDS:
            print("✅ Zvuk detekován, posílám...")
            sender.send_wav(audio)
            last_detection_time = current_time
        else:
            print("...")

except KeyboardInterrupt:
    print("\n🛑 Ukončeno.")