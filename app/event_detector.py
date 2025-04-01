import numpy as np

class AudioDetector:
    def __init__(self, rate=44100, frame_length=1024, hop_length=256, threshold=0.0011):
        self.rate = rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold = threshold

    def detect_rms_impulse(self, audio):
        rms = self.compute_rms_framewise(audio)
        if np.max(rms) < self.threshold:
            return None

        above = rms > self.threshold
        start = np.argmax(above)
        end = len(rms) - np.argmax(above[::-1])
        return (start, end, rms)

    def compute_rms_framewise(self, y):
        return np.array([
            np.sqrt(np.mean(y[i:i + self.frame_length]**2))
            for i in range(0, len(y) - self.frame_length, self.hop_length)
        ])
