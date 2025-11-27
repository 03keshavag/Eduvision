import cv2
from gesture.hand_detector import HandDetector
from gesture.value_processor import ValueProcessor
from gesture.projectile_simulator import ProjectileSimulator

detector = HandDetector()
velocity_processor = ValueProcessor(scale_factor=300, smoothing=0.2)
angle_processor = ValueProcessor(scale_factor=90, smoothing=0.3)

simulator = ProjectileSimulator()

cap = cv2.VideoCapture(0)

velocity = 0
angle = 45

pinch_state = False  # gesture firing flag
PINCH_THRESHOLD = 0.04
RELEASE_THRESHOLD = 0.09

while True:
    success, frame = cap.read()
    if not success:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    left_dist = None
    right_dist = None

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

            detector.draw(frame, hand_landmarks)
            dist = detector.finger_distance(hand_landmarks)

            if handedness.classification[0].label == "Left":
                left_dist = dist
            else:
                right_dist = dist

    # Process live values
    if left_dist is not None:
        angle = max(0, min(90, angle_processor.scale(left_dist)))

    if right_dist is not None:
        velocity = max(0, velocity_processor.scale(right_dist))

        # Gesture firing detection
        if right_dist < PINCH_THRESHOLD:
            pinch_state = True
        elif right_dist > RELEASE_THRESHOLD and pinch_state:
            simulator.launch(velocity, angle)
            print(f"🚀 Gesture Launch | V={velocity} | θ={angle}")
            pinch_state = False

    # --- KEYBOARD LAUNCH SUPPORT ---
    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        simulator.launch(velocity, angle)
        print(f"🚀 Keyboard Launch | V={velocity} | θ={angle}")

    if key & 0xFF == ord("q"):
        break

    # Display info
    cv2.putText(frame, f"Velocity (Right): {round(velocity,2)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Angle (Left): {round(angle,2)} deg", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 180, 0), 2)

    cv2.putText(frame, "Gesture: Pinch+Release OR Press [SPACE] to Launch", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    cv2.imshow("Gesture Control", frame)

    # Projectile view
    canvas = simulator.update()
    cv2.imshow("Projectile Motion", canvas)

cap.release()
cv2.destroyAllWindows()
