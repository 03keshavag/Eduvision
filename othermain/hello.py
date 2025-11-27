import cv2
import mediapipe as mp
import numpy as np
import math
import time


# -------------------------------- HAND TRACKING --------------------------------
class HandDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        left, right = None, None

        if result.multi_hand_landmarks and result.multi_handedness:
            for lm, hand in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = hand.classification[0].label
                self.drawer.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

                # Thumb → Index tip distance
                t1 = lm.landmark[4]
                t2 = lm.landmark[8]
                dist = math.sqrt((t2.x - t1.x) ** 2 + (t2.y - t1.y) ** 2)

                if label == "Left":
                    left = dist
                else:
                    right = dist

        return frame, left, right


# -------------------------------- SMOOTH VALUE SCALER --------------------------------
class ValueProcessor:
    def __init__(self, min_val, max_val, smooth=0.3):
        self.min = min_val
        self.max = max_val
        self.smooth = smooth
        self.last = (min_val + max_val) / 2

    def scale(self, dist):
        dist = max(0.01, min(dist, 0.25))
        norm = (dist - 0.01) / (0.25 - 0.01)
        raw = self.min + norm * (self.max - self.min)
        filtered = self.last * (1 - self.smooth) + raw * self.smooth
        self.last = filtered
        return round(filtered, 2)


# -------------------------------- PROJECTILE ENGINE --------------------------------
class ProjectileEngine:
    def __init__(self):
        self.w, self.h = 1000, 600
        self.reset()

        self.gravity_modes = {"Earth": 9.8, "Moon": 1.6, "Mars": 3.7}
        self.current_gravity = "Earth"

    def reset(self):
        self.t = 0
        self.active = False
        self.path = []

    def launch(self, v, angle):
        self.reset()
        self.v = v
        self.angle = angle
        self.active = True
        print(f"🚀 Launch → V={v} | A={angle}° | Gravity={self.current_gravity}")

    def update(self):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if not self.active:
            return canvas

        g = self.gravity_modes[self.current_gravity]
        rad = math.radians(self.angle)

        x = int(self.v * math.cos(rad) * self.t) + 50  # start from left
        y = int(self.v * math.sin(rad) * self.t - 0.5 * g * self.t ** 2)
        y = self.h - y

        if x >= self.w or y >= self.h or y <= 0:
            self.reset()
            return canvas

        self.path.append((x, y))

        for p in self.path:
            cv2.circle(canvas, p, 10, (0, 255, 255), -1)

        self.t += 0.04
        return canvas

    def predict_path(self, v, angle):
        pts = []
        g = self.gravity_modes[self.current_gravity]
        rad = math.radians(angle)

        for t in np.arange(0, 5, 0.03):
            x = int(v * math.cos(rad) * t) + 50
            y = int(v * math.sin(rad) * t - 0.5 * g * t ** 2)
            y = self.h - y
            if 0 < x < self.w and 0 < y < self.h:
                pts.append((x, y))
        return pts


# ----------------------------- MAIN LOOP -----------------------------
detector = HandDetector()
angle_map = ValueProcessor(30, 90)
velocity_map = ValueProcessor(50, 100)
engine = ProjectileEngine()

cap = cv2.VideoCapture(0)

pinch_ready = False
PINCH = 0.02
RELEASE = 0.07

velocity, angle = 60, 45


while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame, left, right = detector.detect(frame)

    if left:
        angle = angle_map.scale(left)

    if right:
        velocity = velocity_map.scale(right)

        if right < PINCH:
            pinch_ready = True
        elif right > RELEASE and pinch_ready:
            engine.launch(velocity, angle)
            pinch_ready = False

    # HUD
    cv2.putText(frame, f"Velocity: {velocity:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,180,0), 2)

    cv2.imshow("Gesture Control", frame)

    canvas = engine.update()

    if not engine.active:
        for p in engine.predict_path(velocity, angle):
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)

    cv2.imshow("Projectile Motion", canvas)

    # 🔥 FIX: Keyboard must read AFTER both windows
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        engine.launch(velocity, angle)

    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
