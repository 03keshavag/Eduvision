import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_rgb):
        return self.hands.process(frame_rgb)

    def draw(self, frame, hand_landmarks):
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    def finger_distance(self, hand_landmarks):
        """Distance between thumb tip and index tip"""
        x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
        x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
