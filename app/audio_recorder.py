import sounddevice as sd
import config


class AudioRecorder:
    def __init__(self, rate=config.SAMPLE_RATE, channels=config.CHANNELS):
        self.rate = rate
        self.channels = channels


    def record_audio(self, duration):
        audio = sd.rec(int(duration * self.rate), samplerate=self.rate, channels=self.channels, dtype='float32')
        sd.wait()
        return audio.flatten()