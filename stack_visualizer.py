# stack_visualizer_gesture.py
# Gesture-controlled Stack Visualizer using cvzone (HandDetector)
# Controls:
#  - Pinch (thumb + index close) -> PUSH (value mapped from hand vertical position)
#  - Fist (no fingers up) -> POP
#  - 'r' -> reset stack
#  - 'Esc' -> exit

import time
import math
import cv2
import numpy as np

# try import cvzone HandDetector
try:
    from cvzone.HandTrackingModule import HandDetector
except Exception as e:
    print("ERROR: cvzone.HandDetector import failed:", e)
    print("Install dependencies: pip install cvzone opencv-python numpy")
    raise

# Window / canvas sizes
W, H = 640, 720
WINDOW = "Stack Visualizer (Gesture)"
CAM_PREVIEW_W, CAM_PREVIEW_H = 320, 240

# Stack config
stack = []
MAX_SIZE = 8

# Animation state
anim = {"type": None, "value": None, "y": 0.0, "start": 0.0}

# Gesture cooldowns (seconds)
PUSH_COOLDOWN = 0.8
POP_COOLDOWN = 0.8
last_push_time = 0.0
last_pop_time = 0.0

# Camera & detector
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = HandDetector(maxHands=1, detectionCon=0.65)

# drawing helpers
def draw_scene(preview_frame=None):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)

    # Title & instructions
    cv2.putText(img, "Gesture Stack Visualizer", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230,230,230), 2)
    cv2.putText(img, "Pinch -> PUSH (hand height chooses value). Fist -> POP. r -> reset. Esc -> exit", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,190,190), 1)

    # Draw stack slots
    box_w, box_h = 260, 64
    cx = W // 2
    base_y = H - 80
    for i in range(MAX_SIZE):
        y = base_y - i * (box_h + 12)
        cv2.rectangle(img, (cx - box_w // 2, y - box_h), (cx + box_w // 2, y), (60, 60, 80), 2)

    # Draw items in stack
    for idx, val in enumerate(stack):
        y = base_y - idx * (box_h + 12)
        color = (90, 200, 140) if idx != len(stack) - 1 else (80, 160, 250)  # top colored differently
        cv2.rectangle(img, (cx - box_w // 2 + 6, y - box_h + 6), (cx + box_w // 2 - 6, y - 6), color, -1)
        cv2.putText(img, str(val), (cx - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)

    # Draw animation if active (incoming or outgoing box)
    if anim["type"] == "push":
        # animate incoming box descending into top slot
        top_idx = len(stack)
        y_target = base_y - top_idx * (box_h + 12)
        y_pos = int(anim["y"])
        cv2.rectangle(img, (cx - box_w // 2 + 6, y_pos - box_h + 6), (cx + box_w // 2 - 6, y_pos - 6), (50, 220, 100), -1)
        cv2.putText(img, str(anim["value"]), (cx - 20, y_pos - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    elif anim["type"] == "pop":
        # animate top popping up
        top_idx = len(stack) - 1
        if top_idx >= 0:
            y_start = base_y - top_idx * (box_h + 12)
            y_pos = int(anim["y"])
            cv2.rectangle(img, (cx - box_w // 2 + 6, y_pos - box_h + 6), (cx + box_w // 2 - 6, y_pos - 6), (200, 100, 80), -1)
            cv2.putText(img, str(anim["value"]), (cx - 20, y_pos - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)

    # draw preview camera frame (top-right)
    if preview_frame is not None:
        try:
            ph = CAM_PREVIEW_H
            pw = CAM_PREVIEW_W
            # resize and place
            small = cv2.resize(preview_frame, (pw, ph))
            img[8:8 + ph, W - pw - 8:W - 8] = small
            # border
            cv2.rectangle(img, (W - pw - 10, 6), (W - 6, 8 + ph), (255, 255, 255), 1)
        except Exception:
            pass

    # HUD debug
    cv2.putText(img, f"Stack size: {len(stack)}/{MAX_SIZE}", (12, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
    return img

# gesture utilities
def compute_pinch_value(lmList, frame_w, frame_h):
    # lmList items: [x,y,z] pixels from cvzone
    try:
        tx, ty = lmList[4][0], lmList[4][1]   # thumb tip
        ix, iy = lmList[8][0], lmList[8][1]   # index tip
        d = math.hypot(tx - ix, ty - iy)
        diag = math.hypot(frame_w, frame_h)
        norm = min(1.0, d / diag * 3.0)
        pinch_strength = 1.0 - norm  # 1.0 => very close, 0 => far
        return pinch_strength, (tx, ty), (ix, iy)
    except Exception:
        return 0.0, None, None

def hand_y_to_value(ny):
    # ny is normalized 0..1 (0 top), map to 1..999 (higher hand -> larger value)
    try:
        v = int((1.0 - ny) * 998) + 1
        return max(1, min(999, v))
    except Exception:
        return 42

# event triggers
def trigger_push(value):
    global anim, last_push_time
    if len(stack) >= MAX_SIZE: 
        print("[info] stack full, cannot push")
        return
    # start push animation from top of screen to target
    anim["type"] = "push"
    anim["value"] = value
    anim["y"] = 40  # start near top
    anim["start"] = time.time()
    last_push_time = time.time()
    print(f"[action] PUSH {value}")

def trigger_pop():
    global anim, last_pop_time
    if not stack:
        print("[info] stack empty, cannot pop")
        return
    # start pop: animate top moving upward
    anim["type"] = "pop"
    anim["value"] = stack[-1]
    top_idx = len(stack) - 1
    base_y = H - 80
    box_h = 64
    anim["y"] = base_y - top_idx * (box_h + 12)
    anim["start"] = time.time()
    last_pop_time = time.time()
    print(f"[action] POP -> {anim['value']}")

# main loop
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, W, H)

print("[info] Starting gesture-controlled stack. Pinch to push, fist to pop.")
time.sleep(0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera failure")
        break

    # detect hand
    hands = detector.findHands(frame, draw=True)  # hands is list of dicts (if draw=True frame is annotated)
    preview = frame.copy()

    g_hands = []
    if hands:
        # cvzone returns list of hands; each hand is a dict: {'lmList':..., 'center':..., ...}
        for h in hands:
            lmList = h["lmList"]  # list of [x,y,z]
            cx, cy = h["center"]
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            nx = cx / frame_w
            ny = cy / frame_h

            fingersList = None
            try:
                f_list = detector.fingersUp(h)  # list like [0/1,...]
                fingers_count = sum(1 for x in f_list if x == 1)
                fingersList = f_list
            except Exception:
                fingers_count = None

            pinch_strength, thumb_pt, idx_pt = compute_pinch_value(lmList, frame_w, frame_h)
            g_hands.append({
                "nx": nx, "ny": ny, "fingers": fingers_count, "pinch": pinch_strength,
                "lmList": lmList, "raw": h
            })

    # process gesture -> actions with cooldowns and debouncing
    now = time.time()
    if g_hands:
        primary = g_hands[0]
        # POP if fist (no fingers up)
        if primary.get("fingers") is not None:
            if primary["fingers"] == 0 and (now - last_pop_time) > POP_COOLDOWN:
                # do pop
                trigger_pop()

        # PUSH on pinch (pinch_strength close to 1 means fingers together)
        pinch_thresh = 0.78
        if primary.get("pinch") is not None:
            if primary["pinch"] > pinch_thresh and (now - last_push_time) > PUSH_COOLDOWN:
                # compute value from hand vertical normalized position
                value = hand_y_to_value(primary["ny"])
                trigger_push(value)

    # animation step: update anim positions & finish events
    if anim["type"] == "push":
        # move downwards to target slot
        elapsed = time.time() - anim["start"]
        # speed tuned to look good
        anim["y"] += elapsed * 700.0 * 0.02
        anim["start"] = time.time()
        # compute final target Y for top slot
        top_idx = len(stack)
        base_y = H - 80
        box_h = 64
        y_target = base_y - top_idx * (box_h + 12)
        if anim["y"] >= y_target:
            # arrived
            stack.append(anim["value"])
            anim["type"] = None
    elif anim["type"] == "pop":
        # animate upward and then remove top
        elapsed = time.time() - anim["start"]
        anim["y"] -= elapsed * 700.0 * 0.02
        anim["start"] = time.time()
        if anim["y"] < 20:
            # finish pop
            if stack:
                popped = stack.pop()
                # optional: print popped value
                print("[popped]", popped)
            anim["type"] = None

    # draw scene and show preview
    scene = draw_scene(preview_frame=preview)
    cv2.imshow(WINDOW, scene)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        stack.clear()
        anim["type"] = None

# cleanup
cap.release()
cv2.destroyAllWindows()