# gesture_projectile_full.py
"""
Gesture + Projectile demo (single-file)

- MediaPipe Hands for gestures (thumb tip (4) to index tip (8))
- Left hand -> angle (0..90 deg)
- Right hand -> velocity (scaled)
- Flash green directional indicator when gestures change
- Pinch (right) -> release launches projectile
- Two windows: Gesture Camera & Projectile Motion (axes shown)
"""

import time
import math
import cv2
import numpy as np
import mediapipe as mp

# ------------------------------
# Projectile simulator (fixed-scale, axes)
# ------------------------------
class ProjectileSimulator:
    def __init__(self, width=640, height=480, gravity=9.81, pixels_per_meter=12.0):
        self.width = int(width)
        self.height = int(height)
        self.gravity = gravity
        self.pixels_per_meter = float(pixels_per_meter)

        # origin in pixels (left bottom offset)
        self.origin_px = (60, self.height - 60)

        # physics state
        self.projectiles = []   # dicts with x,y,vx,vy
        self.trajectory = []    # (x_m, y_m) samples

        # indicator (gesture-triggered flash)
        self._indicator_angle_deg = 45.0
        self._indicator_velocity = 0.0
        self._indicator_frames_remaining = 0

        # rendering colors
        self.bg_color = (10, 10, 30)
        self.axis_color = (200, 200, 200)
        self.grid_color = (40, 40, 60)
        self.trail_color = (200, 200, 200)
        self.projectile_color = (0, 180, 255)
        self.indicator_color = (0, 255, 0)

        # timekeeping
        self.last_update = time.time()
        self.max_dt = 0.05

        # axis settings: meters shown on X,Y
        self.x_meters_visible = 20.0
        self.y_meters_visible = 10.0

    def set_canvas_size(self, w, h):
        self.width = int(w); self.height = int(h)
        self.origin_px = (60, self.height - 60)

    def set_scale_pixels_per_meter(self, ppm):
        self.pixels_per_meter = float(ppm)

    def launch(self, velocity, angle_deg):
        """Add a projectile launched from world origin (0,0). velocity in m/s"""
        angle_rad = math.radians(angle_deg)
        vx = float(velocity) * math.cos(angle_rad)
        vy = float(velocity) * math.sin(angle_rad)
        self.projectiles.append({'x': 0.0, 'y': 0.0, 'vx': vx, 'vy': vy, 't': time.time()})
        # clear trajectory to draw new path
        self.trajectory = []

    def flash_indicator(self, angle_deg, velocity, frames=4):
        """Show indicator for N frames (call when gesture changes)."""
        self._indicator_angle_deg = float(angle_deg)
        self._indicator_velocity = float(velocity)
        self._indicator_frames_remaining = max(1, int(frames))

    def update(self):
        """Step physics and render BGR canvas (numpy array)."""
        now = time.time()
        dt = now - self.last_update
        dt = max(0.0, min(self.max_dt, dt))
        self.last_update = now

        # Integrate physics (Euler)
        alive = []
        for p in self.projectiles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] -= self.gravity * dt
            # sample for trail
            self.trajectory.append((p['x'], p['y']))
            # keep if not far below ground
            if p['y'] >= -5.0 and p['x'] < 2000.0:
                alive.append(p)
        self.projectiles = alive

        # Create canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # Draw axis rectangle area (right side) or fill whole canvas - we'll draw axes across canvas
        # draw grid lines (every 1 meter)
        self._draw_grid(canvas)

        # draw axes (X and Y with ticks and labels)
        self._draw_axes(canvas)

        # draw trajectory (transform world->canvas)
        if len(self.trajectory) >= 2:
            pts = [self._world_to_canvas(pt) for pt in self.trajectory]
            # connect valid successive points
            for i in range(1, len(pts)):
                if self._pt_ok(pts[i-1]) and self._pt_ok(pts[i]):
                    cv2.line(canvas, pts[i-1], pts[i], self.trail_color, 1)

        # draw current projectiles
        for p in self.projectiles:
            px = self._world_to_canvas((p['x'], p['y']))
            if self._pt_ok(px):
                cv2.circle(canvas, px, 6, self.projectile_color, -1)

        # draw origin marker
        cv2.circle(canvas, self.origin_px, 4, (220,220,220), -1)
        cv2.putText(canvas, "(0,0)m", (self.origin_px[0]+6, self.origin_px[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.axis_color, 1)

        # draw indicator only if frames remaining
        if self._indicator_frames_remaining > 0:
            self._draw_indicator(canvas)
            self._indicator_frames_remaining -= 1

        return canvas

    # ---------------- internal render helpers ----------------
    def _world_to_canvas(self, world_pt):
        """World meters -> canvas px. origin (0,0) placed at self.origin_px.
           world x right, world y up => canvas y decreases upward"""
        x_m, y_m = world_pt
        x_px = int(round(self.origin_px[0] + x_m * self.pixels_per_meter))
        y_px = int(round(self.origin_px[1] - y_m * self.pixels_per_meter))
        return (x_px, y_px)

    def _pt_ok(self, pt):
        if pt is None: return False
        x,y = pt
        return -200 <= x <= self.width+200 and -200 <= y <= self.height+200

    def _draw_grid(self, canvas):
        """Light grid every 1 meter within visible meters using pixels_per_meter"""
        if self.pixels_per_meter <= 0: return
        # Vertical grid lines (x)
        max_x_px = int(self.origin_px[0] + self.x_meters_visible * self.pixels_per_meter)
        for m in range(0, int(self.x_meters_visible)+1):
            x_px = int(self.origin_px[0] + m * self.pixels_per_meter)
            cv2.line(canvas, (x_px, 0), (x_px, self.height), self.grid_color, 1)
        # Horizontal grid lines (y)
        for m in range(0, int(self.y_meters_visible)+1):
            y_px = int(self.origin_px[1] - m * self.pixels_per_meter)
            cv2.line(canvas, (0, y_px), (self.width, y_px), self.grid_color, 1)

    def _draw_axes(self, canvas):
        # X axis: horizontal at origin y. from origin_px[0] -> origin_px[0]+x_meters_visible*ppm
        x_start = self.origin_px[0]
        x_end = int(self.origin_px[0] + self.x_meters_visible * self.pixels_per_meter)
        y_axis = self.origin_px[1]
        cv2.line(canvas, (x_start, y_axis), (x_end, y_axis), self.axis_color, 2)
        # ticks & labels on X
        for m in range(0, int(self.x_meters_visible)+1):
            x_px = int(self.origin_px[0] + m * self.pixels_per_meter)
            cv2.line(canvas, (x_px, y_axis), (x_px, y_axis+8), self.axis_color, 1)
            cv2.putText(canvas, f"{m}m", (x_px-10, y_axis+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.axis_color, 1)

        # Y axis: vertical at origin x (upwards)
        y_top = int(self.origin_px[1] - self.y_meters_visible * self.pixels_per_meter)
        cv2.line(canvas, (self.origin_px[0], self.origin_px[1]), (self.origin_px[0], y_top), self.axis_color, 2)
        # ticks & labels on Y
        for m in range(0, int(self.y_meters_visible)+1):
            y_px = int(self.origin_px[1] - m * self.pixels_per_meter)
            cv2.line(canvas, (self.origin_px[0]-8, y_px), (self.origin_px[0], y_px), self.axis_color, 1)
            cv2.putText(canvas, f"{m}m", (self.origin_px[0]-48, y_px+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.axis_color, 1)

        # Axis labels
        cv2.putText(canvas, "X (m)", (x_end-40, y_axis+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.axis_color, 1)
        cv2.putText(canvas, "Y (m)", (self.origin_px[0]-48, y_top+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.axis_color, 1)

    def _draw_indicator(self, canvas):
        angle_rad = math.radians(self._indicator_angle_deg)
        # length in meters proportional to velocity
        length_m = max(0.05, self._indicator_velocity * 0.05)
        end_world = (length_m * math.cos(angle_rad), length_m * math.sin(angle_rad))
        start_px = self._world_to_canvas((0.0, 0.0))
        end_px = self._world_to_canvas(end_world)
        cv2.line(canvas, start_px, end_px, self.indicator_color, 3)
        self._draw_arrowhead(canvas, start_px, end_px, self.indicator_color)
        cv2.putText(canvas, f"V={round(self._indicator_velocity,2)} m/s  θ={int(self._indicator_angle_deg)}°",
                    (start_px[0]+8, start_px[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.indicator_color, 1)

    def _draw_arrowhead(self, img, p1, p2, color):
        dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
        if dx==0 and dy==0: return
        ang = math.atan2(dy, dx); size = 8
        pA = (int(p2[0] - size*math.cos(ang - math.pi/6)), int(p2[1] - size*math.sin(ang - math.pi/6)))
        pB = (int(p2[0] - size*math.cos(ang + math.pi/6)), int(p2[1] - size*math.sin(ang + math.pi/6)))
        cv2.line(img, p2, pA, color, 2); cv2.line(img, p2, pB, color, 2)


# ------------------------------
# Helper functions for gestures
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalized_distance(lm1, lm2, img_w, img_h):
    x1, y1 = lm1.x * img_w, lm1.y * img_h
    x2, y2 = lm2.x * img_w, lm2.y * img_h
    d = math.hypot(x2 - x1, y2 - y1)
    denom = max(img_w, img_h)
    return d / denom

def smooth(prev, new, alpha=0.25):
    if prev is None: return new
    return prev * (1 - alpha) + new * alpha

# ------------------------------
# Main loop
# ------------------------------
def main():
    # Try to open the camera robustly
    def try_capture(idx):
        # prefer CAP_DSHOW on Windows if available
        if cv2.__name__ and (cv2.__version__):
            # attempt with CAP_DSHOW flag on Windows
            try:
                if cv2.getBuildInformation().lower().find('msvc') >= 0:
                    return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            except Exception:
                pass
        # fallback:
        return cv2.VideoCapture(idx)

    cap = try_capture(0)
    if not cap.isOpened():
        # try other indices
        for i in range(1,4):
            cap = try_capture(i)
            if cap.isOpened():
                break

    if not cap.isOpened():
        print("ERROR: Could not open webcam (tried indices 0..3). Close other apps and try again.")
        return

    # read initial frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Camera opened but frame not read.")
        cap.release()
        return

    cam_h, cam_w = frame.shape[:2]

    # instantiate simulator and match resolution to camera or whichever you prefer
    sim_w, sim_h = 640, 480
    sim = ProjectileSimulator(width=sim_w, height=sim_h, pixels_per_meter=12.0)
    sim.set_canvas_size(sim_w, sim_h)

    # processing state
    velocity = 0.0
    angle = 45.0
    smoothed_velocity = None
    smoothed_angle = None
    prev_left_dist = None
    prev_right_dist = None
    pinch_state = False
    PINCH_THRESHOLD = 0.04
    RELEASE_THRESHOLD = 0.09
    VELOCITY_SCALE = 300.0   # maps normalized distance to "m/s"
    ANGLE_SCALE = 90.0
    DELTA_CHANGE = 0.01

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # mirror view
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            left_dist = None
            right_dist = None

            # draw and compute distances
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    lm = hand_landmarks.landmark
                    d = normalized_distance(lm[4], lm[8], img_w, img_h)  # thumb tip (4) and index tip (8)
                    # debug visual between tips
                    x1, y1 = int(lm[4].x * img_w), int(lm[4].y * img_h)
                    x2, y2 = int(lm[8].x * img_w), int(lm[8].y * img_h)
                    cv2.line(frame, (x1,y1), (x2,y2), (200,200,0), 2)
                    mid = ((x1+x2)//2, (y1+y2)//2)
                    cv2.circle(frame, mid, 6, (255,180,0), -1)

                    if label == "Left":
                        left_dist = d
                    else:
                        right_dist = d

            # map left_dist -> angle
            if left_dist is not None:
                raw_angle = max(0.0, min(1.0, left_dist)) * ANGLE_SCALE
                smoothed_angle = smooth(smoothed_angle, raw_angle, alpha=0.25)
                angle = smoothed_angle

            # map right_dist -> velocity
            if right_dist is not None:
                raw_vel = max(0.0, right_dist) * VELOCITY_SCALE
                smoothed_velocity = smooth(smoothed_velocity, raw_vel, alpha=0.2)
                velocity = smoothed_velocity

                # pinch detection
                if right_dist < PINCH_THRESHOLD:
                    if not pinch_state:
                        pinch_state = True
                        # optional immediate feedback on pinch start:
                        sim.flash_indicator(angle, velocity, frames=4)
                elif right_dist > RELEASE_THRESHOLD and pinch_state:
                    sim.launch(velocity, angle)
                    print(f"🚀 Gesture Launch | V={round(velocity,2)} | θ={round(angle,2)}")
                    pinch_state = False

            # gesture-change detection -> flash indicator only on changes
            if left_dist is not None:
                if prev_left_dist is None or abs(left_dist - prev_left_dist) > DELTA_CHANGE:
                    sim.flash_indicator(angle, velocity, frames=5)
                prev_left_dist = left_dist
            else:
                prev_left_dist = None

            if right_dist is not None:
                if prev_right_dist is None or abs(right_dist - prev_right_dist) > DELTA_CHANGE:
                    sim.flash_indicator(angle, velocity, frames=5)
                prev_right_dist = right_dist
            else:
                prev_right_dist = None

            # keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # space
                sim.launch(velocity, angle)
                print(f"🚀 Keyboard Launch | V={round(velocity,2)} | θ={round(angle,2)}")
            if key == ord('q'):
                break

            # Overlay readouts on camera frame
            cv2.putText(frame, f"Velocity (Right): {0 if velocity is None else round(velocity,2)} m/s", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Angle (Left): {0 if angle is None else round(angle,2)} deg", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,0), 2)
            cv2.putText(frame, "Pinch (Right) to Launch | SPACE to launch | q to quit", (10, img_h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Gesture Camera", frame)

            # Update and show projectile view
            canvas = sim.update()
            cv2.imshow("Projectile Motion (axes)", canvas)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
