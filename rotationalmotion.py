# gesture_motion_combo_with_ui.py
"""
Gesture-controlled Projectile + Rotational Motion demo with clickable UI buttons.

Run:
    python gesture_motion_combo_with_ui.py

Controls:
 - Click "Projectile" or "Rotational" buttons in the Gesture Camera window to switch modes.
 - Or press 'p' (projectile) / 'r' (rotational') on keyboard.
 - SPACE launches in projectile mode.
 - q quits.
"""
import time, math
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------
# Gesture helpers
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalized_distance(lm1, lm2, img_w, img_h):
    x1, y1 = lm1.x * img_w, lm1.y * img_h
    x2, y2 = lm2.x * img_w, lm2.y * img_h
    d = math.hypot(x2 - x1, y2 - y1)
    denom = max(img_w, img_h) if max(img_w, img_h) > 0 else 1.0
    return d / denom

def smooth(prev, new, alpha=0.25):
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha

# ----------------------------
# Projectile simulator (kept simple and stable)
# ----------------------------
class ProjectileSimulator:
    def __init__(self, width=800, height=480, gravity=9.81, pixels_per_meter=12.0):
        self.width = int(width); self.height = int(height)
        self.gravity = gravity
        self.pixels_per_meter = float(pixels_per_meter)
        self.origin_px = (60, self.height - 60)
        self.projectiles = []
        self.trajectory = []
        self._indicator_angle_deg = 45.0
        self._indicator_velocity = 0.0
        self._indicator_frames_remaining = 0
        self.bg_color = (10,10,30)
        self.axis_color = (200,200,200)
        self.grid_color = (40,40,60)
        self.trail_color = (200,200,200)
        self.projectile_color = (0,180,255)
        self.indicator_color = (0,255,0)
        self.last_update = time.time()
        self.max_dt = 0.05

    def set_canvas_size(self,w,h):
        self.width = int(w); self.height = int(h)
        self.origin_px = (60, self.height - 60)

    def set_scale(self, ppm):
        self.pixels_per_meter = float(ppm)

    def launch(self, velocity, angle_deg):
        angle_rad = math.radians(angle_deg)
        vx = float(velocity) * math.cos(angle_rad)
        vy = float(velocity) * math.sin(angle_rad)
        self.projectiles.append({'x':0.0,'y':0.0,'vx':vx,'vy':vy,'t':time.time()})
        self.trajectory = []

    def flash_indicator(self, angle_deg, velocity, frames=4):
        self._indicator_angle_deg = float(angle_deg)
        self._indicator_velocity = float(velocity)
        self._indicator_frames_remaining = max(1,int(frames))

    def update(self):
        now = time.time()
        dt = now - self.last_update
        dt = max(0.0, min(self.max_dt, dt))
        self.last_update = now

        alive = []
        for p in self.projectiles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] -= self.gravity * dt
            self.trajectory.append((p['x'], p['y']))
            if p['y'] >= -10.0 and p['x'] < 2000.0:
                alive.append(p)
        self.projectiles = alive

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # simple grid & axes (0..20m x, 0..10m y)
        if self.pixels_per_meter > 0:
            for m in range(0, 21):
                x_px = int(self.origin_px[0] + m * self.pixels_per_meter)
                if 0 <= x_px <= self.width:
                    cv2.line(canvas, (x_px, 0), (x_px, self.height), self.grid_color, 1)
            for m in range(0, 11):
                y_px = int(self.origin_px[1] - m * self.pixels_per_meter)
                if 0 <= y_px <= self.height:
                    cv2.line(canvas, (0, y_px), (self.width, y_px), self.grid_color, 1)

        # axes + ticks
        x_end = int(self.origin_px[0] + 20 * self.pixels_per_meter)
        cv2.line(canvas, (self.origin_px[0], self.origin_px[1]), (x_end, self.origin_px[1]), self.axis_color, 2)
        cv2.line(canvas, (self.origin_px[0], self.origin_px[1]), (self.origin_px[0], int(self.origin_px[1] - 10 * self.pixels_per_meter)), self.axis_color, 2)
        for m in range(0, 21):
            x_px = int(self.origin_px[0] + m * self.pixels_per_meter)
            cv2.line(canvas, (x_px, self.origin_px[1]), (x_px, self.origin_px[1] + 6), self.axis_color, 1)
            if 0 <= x_px <= self.width:
                cv2.putText(canvas, f"{m}m", (x_px-12, self.origin_px[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.axis_color, 1)
        for m in range(0, 11):
            y_px = int(self.origin_px[1] - m * self.pixels_per_meter)
            cv2.line(canvas, (self.origin_px[0]-6, y_px), (self.origin_px[0], y_px), self.axis_color, 1)
            if 0 <= y_px <= self.height:
                cv2.putText(canvas, f"{m}m", (self.origin_px[0]-48, y_px+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.axis_color, 1)

        # trajectory
        if len(self.trajectory) >= 2:
            prev_px = None
            for (wx, wy) in self.trajectory:
                px = self._world_to_canvas((wx, wy))
                if prev_px is not None:
                    cv2.line(canvas, prev_px, px, self.trail_color, 1)
                prev_px = px

        # projectiles
        for p in self.projectiles:
            px = self._world_to_canvas((p['x'], p['y']))
            if self._pt_ok(px):
                cv2.circle(canvas, px, 6, self.projectile_color, -1)

        # origin marker
        cv2.circle(canvas, self.origin_px, 4, (220,220,220), -1)
        cv2.putText(canvas, "(0,0)m", (self.origin_px[0]+6, self.origin_px[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.axis_color, 1)

        # indicator
        if self._indicator_frames_remaining > 0:
            self._draw_indicator(canvas)
            self._indicator_frames_remaining -= 1

        return canvas

    def _world_to_canvas(self, world_pt):
        x_m, y_m = world_pt
        x_px = int(round(self.origin_px[0] + x_m * self.pixels_per_meter))
        y_px = int(round(self.origin_px[1] - y_m * self.pixels_per_meter))
        return (x_px, y_px)

    def _pt_ok(self, pt):
        if pt is None: return False
        x,y = pt
        return -200 <= x <= self.width+200 and -200 <= y <= self.height+200

    def _draw_indicator(self, canvas):
        ang = math.radians(self._indicator_angle_deg)
        length_m = max(0.05, self._indicator_velocity * 0.05)
        end_world = (length_m * math.cos(ang), length_m * math.sin(ang))
        start_px = self._world_to_canvas((0.0, 0.0))
        end_px = self._world_to_canvas(end_world)
        cv2.line(canvas, start_px, end_px, self.indicator_color, 3)
        self._draw_arrowhead(canvas, start_px, end_px, self.indicator_color)
        cv2.putText(canvas, f"V={round(self._indicator_velocity,2)} m/s θ={int(self._indicator_angle_deg)}°", (start_px[0]+8, start_px[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.indicator_color, 1)

    def _draw_arrowhead(self, img, p1, p2, color):
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        if dx == 0 and dy == 0: return
        ang = math.atan2(dy, dx); s = 8
        pA = (int(p2[0] - s * math.cos(ang - math.pi/6)), int(p2[1] - s * math.sin(ang - math.pi/6)))
        pB = (int(p2[0] - s * math.cos(ang + math.pi/6)), int(p2[1] - s * math.sin(ang + math.pi/6)))
        cv2.line(img, p2, pA, color, 2); cv2.line(img, p2, pB, color, 2)

# ----------------------------
# Rotational motion (same as before)
# ----------------------------
class RotationalSimulator:
    def __init__(self, width=800, height=480):
        self.width = int(width); self.height = int(height)
        self.center = (width//2, height//2)
        self.radius_m = 1.0
        self.pixels_per_meter = 80.0
        self.omega = 2.0
        self.theta = 0.0
        self.trail = []
        self.last_update = time.time()
        self.frames_to_flash = 0
        self.flash_angle = 0.0
        self.flash_speed = 0.0

    def set_canvas_size(self,w,h):
        self.width=int(w); self.height=int(h)
        self.center = (self.width//2, self.height//2)

    def set_radius_m(self, r):
        self.radius_m = max(0.05, float(r))

    def set_omega(self, w):
        self.omega = float(w)

    def flash_indicator(self, angle_deg, speed, frames=4):
        self.frames_to_flash = max(1,int(frames))
        self.flash_angle = angle_deg
        self.flash_speed = speed

    def update(self):
        now = time.time()
        dt = now - self.last_update
        dt = max(0.0, min(0.05, dt))
        self.last_update = now

        self.theta += self.omega * dt
        x = self.radius_m * math.cos(self.theta)
        y = self.radius_m * math.sin(self.theta)
        self.trail.append((x, y))
        if len(self.trail) > 400: self.trail.pop(0)

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8); canvas[:] = (20,20,20)
        r_px = int(round(self.radius_m * self.pixels_per_meter))
        cv2.circle(canvas, self.center, abs(r_px), (70,70,70), 2)
        cv2.line(canvas, (0,self.center[1]), (self.width, self.center[1]), (100,100,100), 1)
        cv2.line(canvas, (self.center[0],0), (self.center[0], self.height), (100,100,100), 1)

        pts = [ (int(self.center[0]+xx*self.pixels_per_meter), int(self.center[1]-yy*self.pixels_per_meter)) for (xx,yy) in self.trail ]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], (200,200,200), 1)

        px = (int(self.center[0] + x * self.pixels_per_meter), int(self.center[1] - y * self.pixels_per_meter))
        cv2.circle(canvas, px, 8, (0,180,255), -1)

        if self.frames_to_flash > 0:
            ang = math.radians(self.flash_angle)
            length_m = max(0.05, self.flash_speed * 0.05)
            end_px = (int(self.center[0] + length_m*math.cos(ang)*self.pixels_per_meter),
                      int(self.center[1] - length_m*math.sin(ang)*self.pixels_per_meter))
            cv2.line(canvas, self.center, end_px, (0,255,0), 3)
            cv2.putText(canvas, f"r={round(self.radius_m,2)}m ω={round(self.omega,2)}rad/s", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),1)
            self.frames_to_flash -= 1
        else:
            cv2.putText(canvas, f"r={round(self.radius_m,2)}m ω={round(self.omega,2)}rad/s", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)

        # small time series graph (same as before)
        graph_w = 220; graph_h = 140
        gx = self.width - graph_w - 10; gy = 10
        cv2.rectangle(canvas, (gx,gy), (gx+graph_w, gy+graph_h), (40,40,40), -1)
        N = min(len(self.trail), graph_w-20)
        if N>1:
            segment = self.trail[-N:]
            xs = [s[0] for s in segment]; ys=[s[1] for s in segment]
            for i in range(1, N):
                x1 = gx+10 + (i-1)*(graph_w-20)/(N-1); y1 = gy+graph_h//2 - int((xs[i-1]/self.radius_m)*(graph_h//2-10))
                x2 = gx+10 + i*(graph_w-20)/(N-1);     y2 = gy+graph_h//2 - int((xs[i]/self.radius_m)*(graph_h//2-10))
                cv2.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), (0,180,255), 1)
                x1 = gx+10 + (i-1)*(graph_w-20)/(N-1); y1 = gy+graph_h//2 - int((ys[i-1]/self.radius_m)*(graph_h//2-10))
                x2 = gx+10 + i*(graph_w-20)/(N-1);     y2 = gy+graph_h//2 - int((ys[i]/self.radius_per_meter if False else ys[i]/self.radius_m)*(graph_h//2-10))
                cv2.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1)
        cv2.putText(canvas, "x(t):orange y(t):green", (gx+8, gy+graph_h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200),1)

        return canvas

# ----------------------------
# UI / main loop
# ----------------------------
def main():
    # camera open
    def try_capture(i):
        try:
            if cv2.getBuildInformation().lower().find('msvc') >= 0:
                return cv2.VideoCapture(i, cv2.CAP_DSHOW)
        except Exception:
            pass
        return cv2.VideoCapture(i)

    cap = try_capture(0)
    if not cap.isOpened():
        for i in range(1,4):
            cap = try_capture(i)
            if cap.isOpened(): break
    if not cap.isOpened():
        print("ERROR: Could not open camera. Close other apps and try indices 0..3.")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Camera opened but could not read frame."); cap.release(); return
    cam_h, cam_w = frame.shape[:2]

    # sims
    sim_w, sim_h = 800, 480
    proj = ProjectileSimulator(sim_w, sim_h, pixels_per_meter=12.0)
    rot = RotationalSimulator(sim_w, sim_h)
    proj.set_canvas_size(sim_w, sim_h); rot.set_canvas_size(sim_w, sim_h)

    # mode state as mutable so mouse callback can update it
    state = {'mode': 'projectile', 'clicked': False}

    # UI button positions (in camera window coordinates)
    BUTTON_W, BUTTON_H = 140, 36
    btn1_pos = (10, 10, 10 + BUTTON_W, 10 + BUTTON_H)   # Projectile button rect (x1,y1,x2,y2)
    btn2_pos = (10 + BUTTON_W + 8, 10, 10 + 2*BUTTON_W + 8, 10 + BUTTON_H)  # Rotational

    # mouse callback to toggle mode via clicks on buttons
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        x1,y1,x2,y2 = btn1_pos
        if x1 <= x <= x2 and y1 <= y <= y2:
            state['mode'] = 'projectile'
            state['clicked'] = True
            print("Mode set to PROJECTILE (clicked)")
            return
        x1,y1,x2,y2 = btn2_pos
        if x1 <= x <= x2 and y1 <= y <= y2:
            state['mode'] = 'rotational'
            state['clicked'] = True
            print("Mode set to ROTATIONAL (clicked)")
            return

    cv2.namedWindow("Gesture Camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Gesture Camera", on_mouse)

    # gesture params
    velocity = 0.0; angle = 45.0
    sm_vel = None; sm_ang = None
    prev_left = None; prev_right = None
    pinch_state = False
    PINCH_THRESHOLD = 0.04; RELEASE_THRESHOLD = 0.09
    VELOCITY_SCALE = 300.0; ANGLE_SCALE = 90.0
    DELTA_CHANGE = 0.01

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
            results = hands.process(rgb); rgb.flags.writeable=True

            left_dist = None; right_dist = None
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark
                    d = normalized_distance(lm[4], lm[8], img_w, img_h)
                    x1,y1 = int(lm[4].x*img_w), int(lm[4].y*img_h)
                    x2,y2 = int(lm[8].x*img_w), int(lm[8].y*img_h)
                    cv2.line(frame,(x1,y1),(x2,y2),(200,200,0),2)
                    mid = ((x1+x2)//2, (y1+y2)//2); cv2.circle(frame, mid, 6, (255,180,0), -1)
                    if label == "Left": left_dist = d
                    else: right_dist = d

            # MODE handling
            mode = state['mode']

            if mode == 'projectile':
                if left_dist is not None:
                    raw_angle = max(0.0, min(1.0, left_dist)) * ANGLE_SCALE
                    sm_ang = smooth(sm_ang, raw_angle, alpha=0.25)
                    angle = sm_ang
                if right_dist is not None:
                    raw_vel = max(0.0, right_dist) * VELOCITY_SCALE
                    sm_vel = smooth(sm_vel, raw_vel, alpha=0.2)
                    velocity = sm_vel
                    if right_dist < PINCH_THRESHOLD:
                        if not pinch_state:
                            pinch_state = True
                            proj.flash_indicator(angle, velocity, frames=4)
                    elif right_dist > RELEASE_THRESHOLD and pinch_state:
                        proj.launch(velocity, angle)
                        print(f"🚀 Gesture Launch | V={round(velocity,2)} | θ={round(angle,2)}")
                        pinch_state = False

                if left_dist is not None:
                    if prev_left is None or abs(left_dist - prev_left) > DELTA_CHANGE:
                        proj.flash_indicator(angle, velocity, frames=5)
                    prev_left = left_dist
                else:
                    prev_left = None

                if right_dist is not None:
                    if prev_right is None or abs(right_dist - prev_right) > DELTA_CHANGE:
                        proj.flash_indicator(angle, velocity, frames=5)
                    prev_right = right_dist
                else:
                    prev_right = None

                canvas = proj.update()

            else:  # rotational
                if left_dist is not None:
                    raw_r = max(0.02, min(0.6, left_dist)) * 4.0
                    r_sm = smooth(getattr(rot, 'radius_m', None), raw_r, alpha=0.25)
                    rot.set_radius_m(r_sm)
                if right_dist is not None:
                    raw_w = max(0.0, min(0.6, right_dist)) * 8.0
                    w_sm = smooth(getattr(rot, 'omega', None), raw_w, alpha=0.2)
                    rot.set_omega(w_sm)
                    if right_dist < PINCH_THRESHOLD:
                        rot.flash_indicator(0, w_sm, frames=4)

                if left_dist is not None:
                    if prev_left is None or abs(left_dist - prev_left) > DELTA_CHANGE:
                        rot.flash_indicator(0, rot.omega, frames=5)
                    prev_left = left_dist
                else:
                    prev_left = None

                if right_dist is not None:
                    if prev_right is None or abs(right_dist - prev_right) > DELTA_CHANGE:
                        rot.flash_indicator(0, rot.omega, frames=5)
                    prev_right = right_dist
                else:
                    prev_right = None

                canvas = rot.update()

            # Draw clickable buttons onto camera frame
            # Button background
            bx1, by1, bx2, by2 = btn1_pos
            cv2.rectangle(frame, (bx1,by1), (bx2,by2), (50,50,50), -1)
            cv2.putText(frame, "Projectile", (bx1+8, by1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
            bx1b, by1b, bx2b, by2b = btn2_pos
            cv2.rectangle(frame, (bx1b,by1b), (bx2b,by2b), (50,50,50), -1)
            cv2.putText(frame, "Rotational", (bx1b+8, by1b+24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
            # highlight selected
            if mode == 'projectile':
                cv2.rectangle(frame, (btn1_pos[0]-2, btn1_pos[1]-2), (btn1_pos[2]+2, btn1_pos[3]+2), (0,180,255), 2)
            else:
                cv2.rectangle(frame, (btn2_pos[0]-2, btn2_pos[1]-2), (btn2_pos[2]+2, btn2_pos[3]+2), (0,180,255), 2)

            # overlays
            cv2.putText(frame, f"Mode: {mode}", (10, cam_h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            if mode == 'projectile':
                cv2.putText(frame, f"V (Right): {0 if velocity is None else round(velocity,2)} m/s", (10, cam_h-50), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
                cv2.putText(frame, f"Angle (Left): {0 if angle is None else round(angle,2)} deg", (10, cam_h-30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,180,0),2)
            else:
                cv2.putText(frame, f"r (Left): {round(rot.radius_m,2)} m", (10, cam_h-50), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
                cv2.putText(frame, f"ω (Right): {round(rot.omega,2)} rad/s", (10, cam_h-30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,180,0),2)

            cv2.imshow("Gesture Camera", frame)
            cv2.imshow("Simulator", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                state['mode'] = 'rotational'
            if key == ord('p'):
                state['mode'] = 'projectile'
            if key == 32:  # space
                if state['mode'] == 'projectile':
                    proj.launch(velocity, angle)
                    print(f"🚀 Keyboard Launch | V={round(velocity,2)} | θ={round(angle,2)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
