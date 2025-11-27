import numpy as np
import cv2
import math

class ProjectileSimulator:

    def __init__(self, width=800, height=500, gravity=9.8):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.reset()

    def reset(self):
        self.active = False
        self.time = 0
        self.velocity = 0
        self.angle = 45

    def launch(self, velocity, angle):
        velocity = max(velocity, 5)
        self.velocity = velocity
        self.angle = angle
        self.time = 0
        self.active = True
        print(f"🚀 LAUNCH | V={velocity}, A={angle}")

    def update(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not self.active:
            return canvas

        rad = math.radians(self.angle)

        x = int(self.velocity * math.cos(rad) * self.time)
        y = int(self.velocity * math.sin(rad) * self.time - 0.5 * self.gravity * (self.time ** 2))
        y = self.height - y

        if x >= self.width or y >= self.height or y <= 0:
            self.reset()
            return canvas

        cv2.circle(canvas, (x, y), 14, (0, 255, 255), -1)
        self.time += 0.05

        return canvas

    def get_prediction_path(self, velocity, angle):
        """Returns predicted points for trajectory curve"""
        points = []
        rad = math.radians(angle)

        for t in np.arange(0, 3, 0.1):  # simulate for time
            x = int(velocity * math.cos(rad) * t)
            y = int(velocity * math.sin(rad) * t - 0.5 * self.gravity * (t ** 2))
            y = self.height - y
            if 0 < x < self.width and 0 < y < self.height:
                points.append((x, y))
        return points
