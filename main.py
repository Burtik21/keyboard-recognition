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
        print(audio.dtype, np.min(audio), np.max(audio))
        result = detector.detect_rms_impulse(audio)

        current_time = time.time()

        if result and current_time - last_detection_time > config.COOLDOWN_SECONDS:
            print("✅ Zvuk detekován, posílám...")
            trigger_value = detector.get_triggering_rms_value(audio)
            if trigger_value:
                print(f"🎯 RMS, které spustilo detekci: {trigger_value:.6f}")
            sender.send_wav(audio)
            last_detection_time = current_time
        else:
            print("...")

except KeyboardInterrupt:
    print("\n🛑 Ukončeno.")