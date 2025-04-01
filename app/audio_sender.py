import requests
import io
from scipy.io.wavfile import write
import numpy as np
import config

class AudioSender:
    def __init__(self, rate=44100):
        self.rate = rate

    def send_wav(self, audio_data):
        # Převod float32 [-1,1] na int16 pro WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        final_url = config.BASE_URL + "/upload-audio"
        buffer = io.BytesIO()
        write(buffer, self.rate, audio_int16)
        buffer.seek(0)

        files = {'file': ('clip.wav', buffer, 'audio/wav')}
        try:
            response = requests.post(config.BASE_URL, files=files, timeout=5)
            print("📤 Odesláno! Status:", response.status_code)
            return response
        except Exception as e:
            print("❌ Chyba při odesílání:", e)
            return None
