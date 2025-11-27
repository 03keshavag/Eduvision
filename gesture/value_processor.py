class ValueProcessor:
    def __init__(self, scale_factor=300, smoothing=0.2):
        self.scale_factor = scale_factor
        self.smoothing = smoothing
        self.previous_value = 0.0

    def scale(self, dist):
        raw = dist * self.scale_factor
        smooth = self.smoothing * raw + (1 - self.smoothing) * self.previous_value
        self.previous_value = smooth
        return round(smooth, 2)
