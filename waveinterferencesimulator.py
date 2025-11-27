# wave_interference_explained.py
# Improved wave interference simulator with colorbar and cross-section graph + clearer labels
import time, math
import cv2
import numpy as np
import mediapipe as mp

# ----------------- config ---------------
SIM_W, SIM_H = 780, 480
C = 1.0
DECAY = 0.08
COLORMAP = cv2.COLORMAP_TURBO
FPS_TARGET = 30.0

# gesture mapping
FREQ_MIN, FREQ_MAX = 0.2, 6.0
AMP_MIN, AMP_MAX = 0.1, 1.5
PINCH_THRESHOLD = 0.04
RELEASE_THRESHOLD = 0.10
INDICATOR_FRAMES = 5

# UI buttons
BTN_W, BTN_H = 140, 36
BTN_MARGIN = 10

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalized_distance(lm1, lm2, img_w, img_h):
    x1,y1 = lm1.x*img_w, lm1.y*img_h
    x2,y2 = lm2.x*img_w, lm2.y*img_h
    d = math.hypot(x2-x1, y2-y1)
    denom = max(img_w, img_h) if max(img_w, img_h)>0 else 1.0
    return d/denom

def lerp(a,b,t): return a + (b-a)*t
def map_range(v,a,b,c,d):
    t = (v-a)/(b-a) if b!=a else 0.0
    t = max(0.0, min(1.0, t))
    return lerp(c,d,t)
def smooth(prev,new,alpha=0.25): return new if prev is None else prev*(1-alpha)+new*alpha

# ----------------- Wave simulator class ----------------
class WaveSimulator:
    def __init__(self, w=SIM_W, h=SIM_H, c=C, decay=DECAY, colormap=COLORMAP):
        self.width = int(w); self.height = int(h)
        self.c = float(c); self.decay = float(decay)
        self.colormap = colormap

        # two sources initial positions and params
        self.sources = [
            {'pos': np.array([self.width*0.33, self.height*0.5]), 'amp': 0.9, 'freq': 1.2, 'phase':0.0},
            {'pos': np.array([self.width*0.66, self.height*0.5]), 'amp': 0.9, 'freq': 1.2, 'phase':0.0},
        ]
        self.two_source = True
        self.grid_x, self.grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.time0 = time.time()
        self.t = 0.0
        self.paused = False
        self._indicator_frames = 0
        self._indicator_params = (0.0, 0.0)
        self.clamp_v = 0.9
        self.last_render = None

    def reset(self):
        self.sources[0]['pos'] = np.array([self.width*0.33, self.height*0.5])
        self.sources[1]['pos'] = np.array([self.width*0.66, self.height*0.5])
        self.sources[0]['amp']=0.9; self.sources[1]['amp']=0.9
        self.sources[0]['freq']=1.2; self.sources[1]['freq']=1.2
        self.paused=False

    def toggle_pause(self):
        self.paused = not self.paused

    def set_indicator(self, freq, amp, frames=INDICATOR_FRAMES):
        self._indicator_params = (freq, amp)
        self._indicator_frames = max(1,int(frames))

    def set_single_mode(self, single=True):
        self.two_source = not single

    def add_or_move_source(self, idx, px, py):
        if idx<0 or idx>=len(self.sources): return
        self.sources[idx]['pos'] = np.array([float(px), float(py)])

    def step_time(self, dt):
        if not self.paused:
            self.t += dt

    def compute_field(self):
        H,W = self.height, self.width
        field = np.zeros((H,W), dtype=np.float32)
        active = [self.sources[0]]
        if self.two_source: active.append(self.sources[1])
        for s in active:
            sx, sy = s['pos']
            rx = (self.grid_x - sx).astype(np.float32)
            ry = (self.grid_y - sy).astype(np.float32)
            r = np.hypot(rx, ry) + 1e-6
            omega = 2.0*math.pi*float(s['freq'])
            phase = float(s.get('phase',0.0))
            arg = omega*(self.t - (r/self.c)) + phase
            att = 1.0 / (1.0 + r*self.decay)
            contrib = float(s['amp']) * att * np.sin(arg)
            field += contrib.astype(np.float32)
        return field

    def render_field(self, field):
        vmax = max(0.5, np.max(np.abs(field)))
        self.clamp_v = 0.9*self.clamp_v + 0.1*vmax if getattr(self,'clamp_v',None) else vmax
        vmax = max(0.2, self.clamp_v)
        img_norm = (field / vmax) * 127.0 + 128.0
        img_u8 = np.clip(img_norm, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8, self.colormap)
        return colored, vmax

    def render(self):
        field = self.compute_field()
        canvas, vmax = self.render_field(field)
        canvas = canvas.copy()
        # draw sources markers
        cv2.circle(canvas, tuple(self.sources[0]['pos'].astype(int)), 8, (255,255,255), -1)
        cv2.putText(canvas, f"A={self.sources[0]['amp']:.2f} f={self.sources[0]['freq']:.2f}Hz",
                    (int(self.sources[0]['pos'][0]+10), int(self.sources[0]['pos'][1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)
        if self.two_source:
            cv2.circle(canvas, tuple(self.sources[1]['pos'].astype(int)), 8, (255,255,255), -1)
            cv2.putText(canvas, f"A={self.sources[1]['amp']:.2f} f={self.sources[1]['freq']:.2f}Hz",
                        (int(self.sources[1]['pos'][0]+10), int(self.sources[1]['pos'][1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

        # overlay status & indicator
        if self._indicator_frames>0:
            txt = f"Freq={self._indicator_params[0]:.2f}Hz Amp={self._indicator_params[1]:.2f}"
            cv2.putText(canvas, txt, (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            self._indicator_frames -= 1
        else:
            st = "PAUSED" if self.paused else "RUNNING"
            cv2.putText(canvas, f"t={self.t:.2f}s  {st}", (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240),1)

        # store last render (raw field too) for cross-section plotting
        self.last_render = {'canvas':canvas, 'field':field, 'vmax':vmax}
        return canvas

# -------------------- Main interactive app --------------------
def main():
    def try_capture(i):
        try:
            if cv2.getBuildInformation().lower().find('msvc')>=0:
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
        print("ERROR: could not open camera (0..3)")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: camera opened but frame read failed"); cap.release(); return
    cam_h, cam_w = frame.shape[:2]

    sim = WaveSimulator(w=SIM_W, h=SIM_H, c=C, decay=DECAY, colormap=COLORMAP)
    sim.set_indicator(1.2,0.9,frames=0)

    # buttons on camera window
    btn1 = (BTN_MARGIN, BTN_MARGIN, BTN_MARGIN+BTN_W, BTN_MARGIN+BTN_H)   # single
    btn2 = (BTN_MARGIN+BTN_W+8, BTN_MARGIN, BTN_MARGIN+2*BTN_W+8, BTN_MARGIN+BTN_H)  # two
    play_rect = (cam_w-110, BTN_MARGIN, cam_w-10, BTN_MARGIN+BTN_H)

    prev_left=None; prev_right=None; sm_freq=None; sm_amp=None; pinch_state=False

    # mouse callback: handle buttons or move source (click)
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn1[0]<=x<=btn1[2] and btn1[1]<=y<=btn1[3]:
                sim.two_source = False; print("UI: single source")
                return
            if btn2[0]<=x<=btn2[2] and btn2[1]<=y<=btn2[3]:
                sim.two_source = True; print("UI: two source"); return
            if play_rect[0]<=x<=play_rect[2] and play_rect[1]<=y<=play_rect[3]:
                sim.toggle_pause(); print("UI: toggled pause:", sim.paused); return
            # else: move nearest source (map camera coords to sim coords)
            sx = int((x / cam_w) * sim.width)
            sy = int((y / cam_h) * sim.height)
            d0 = np.hypot(sx - sim.sources[0]['pos'][0], sy - sim.sources[0]['pos'][1])
            d1 = np.hypot(sx - sim.sources[1]['pos'][0], sy - sim.sources[1]['pos'][1])
            idx = 0 if d0<=d1 else 1
            sim.add_or_move_source(idx, sx, sy)
            print(f"Moved source {idx} -> ({sx},{sy})")

    cv2.namedWindow("Gesture Camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Gesture Camera", on_mouse)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    last_time = time.time()
    cursor_y = sim.height // 2   # default cross-section row (use mouse y on camera if we want)
    show_center_line = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue
            frame = cv2.flip(frame,1)
            img_h, img_w = frame.shape[:2]

            # draw UI buttons
            cv2.rectangle(frame, (btn1[0],btn1[1]), (btn1[2],btn1[3]), (40,40,40), -1)
            cv2.putText(frame, "Single", (btn1[0]+16, btn1[1]+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220),2)
            cv2.rectangle(frame, (btn2[0],btn2[1]), (btn2[2],btn2[3]), (40,40,40), -1)
            cv2.putText(frame, "Two", (btn2[0]+56, btn2[1]+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220),2)
            cv2.rectangle(frame, (play_rect[0],play_rect[1]), (play_rect[2],play_rect[3]), (40,40,40), -1)
            cv2.putText(frame, "Play/Pause", (play_rect[0]+6, play_rect[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220),1)

            # process hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
            results = hands.process(rgb); rgb.flags.writeable=True
            left_dist=None; right_dist=None
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark
                    d = normalized_distance(lm[4], lm[8], img_w, img_h)
                    x1,y1 = int(lm[4].x*img_w), int(lm[4].y*img_h)
                    x2,y2 = int(lm[8].x*img_w), int(lm[8].y*img_h)
                    cv2.line(frame, (x1,y1),(x2,y2),(200,200,0),2)
                    mid = ((x1+x2)//2, (y1+y2)//2); cv2.circle(frame, mid,6,(255,180,0),-1)
                    if label == "Left": left_dist = d
                    else: right_dist = d

            # map gestures
            dt = time.time() - last_time; last_time = time.time()
            sim.step_time(dt)
            gesture_changed = False

            # left -> frequency
            if left_dist is not None:
                freq_raw = map_range(left_dist, 0.02, 0.5, FREQ_MIN, FREQ_MAX)
                sm_freq = smooth(sm_freq if 'sm_freq' in locals() else None, freq_raw, alpha=0.2)
                for s in sim.sources:
                    if abs(s['freq'] - sm_freq) > 1e-3: gesture_changed = True
                    s['freq'] = sm_freq

            # right -> amplitude + pinch pause toggle
            if right_dist is not None:
                amp_raw = map_range(right_dist, 0.01, 0.6, AMP_MIN, AMP_MAX)
                sm_amp = smooth(sm_amp if 'sm_amp' in locals() else None, amp_raw, alpha=0.2)
                for s in sim.sources:
                    if abs(s['amp'] - sm_amp) > 1e-3: gesture_changed = True
                    s['amp'] = sm_amp

                if right_dist < PINCH_THRESHOLD:
                    if not pinch_state:
                        pinch_state=True
                        sim.toggle_pause()
                        print("Pinch: toggled pause ->", sim.paused)
                elif right_dist > RELEASE_THRESHOLD and pinch_state:
                    pinch_state=False

            if gesture_changed:
                sim.set_indicator(sim.sources[0]['freq'], sim.sources[0]['amp'], frames=INDICATOR_FRAMES)

            # render simulator canvas and get field
            canvas = sim.render()  # BGR image
            # draw colorbar at right side
            field = sim.last_render['field'] if sim.last_render else None
            vmax = sim.last_render['vmax'] if sim.last_render else 1.0
            # colorbar parameters
            bar_w = 24; gutter = 8
            bar_h = canvas.shape[0]-80
            bx = canvas.shape[1] - bar_w - gutter; by = 40
            # build colorbar image from colormap
            bar = np.linspace(-vmax, vmax, bar_h).astype(np.float32)[::-1]  # top->bottom
            bar_norm = (bar / vmax) * 127.0 + 128.0
            bar_u8 = np.clip(bar_norm,0,255).astype(np.uint8)
            bar_col = cv2.applyColorMap(bar_u8.reshape(-1,1), COLORMAP)
            bar_col = cv2.resize(bar_col, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)
            # paste bar onto canvas
            canvas[by:by+bar_h, bx:bx+bar_w] = bar_col
            # label ticks on colorbar
            n_ticks = 5
            for i in range(n_ticks):
                ty = by + int(i*(bar_h-1)/(n_ticks-1))
                val = lerp(vmax, -vmax, i/(n_ticks-1))
                cv2.putText(canvas, f"{val:.2f}", (bx - 70, ty+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240,240,240),1)

            # cross-section row: use mouse y position mapped from camera mouse (optional)
            # default: center row unless user clicks (we allowed moving sources by click earlier)
            row = sim.height//2
            # sample field row and plot amplitude vs x into a small plot area under canvas
            if field is not None:
                # pick row (center)
                row_vals = field[row, :]
                plot_h = 140; plot_w = canvas.shape[1]
                plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
                plot[:] = (10,10,10)
                # normalize row to ±vmax
                pv = row_vals.copy()
                px = np.linspace(0, plot_w-1, len(pv)).astype(int)
                mid_y = plot_h//2
                for i in range(1, len(px)):
                    y1 = int(mid_y - (pv[i-1]/vmax)*(plot_h//2-6))
                    y2 = int(mid_y - (pv[i]/vmax)*(plot_h//2-6))
                    cv2.line(plot, (px[i-1], y1), (px[i], y2), (0,180,255), 1)
                cv2.putText(plot, "Cross-section (amplitude vs x) at center row", (8,12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220),1)
                # place plot beneath canvas
                combined_h = canvas.shape[0] + plot.shape[0] + 8
                out = np.zeros((combined_h, canvas.shape[1], 3), dtype=np.uint8)
                out[:canvas.shape[0], :, :] = canvas
                out[canvas.shape[0]+8:canvas.shape[0]+8+plot.shape[0], :, :] = plot
            else:
                out = canvas

            # overlay help text on camera
            cv2.putText(frame, "Left hand: frequency | Right hand: amplitude | Pinch: pause", (10, cam_h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
            cv2.putText(frame, "Click simulator window to move sources. Buttons: Single/Two/Play", (10, cam_h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200),1)
            cv2.putText(frame, "Keys: s(single) d(two) c(reset) q(quit)", (10, cam_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200),1)

            # show windows
            cv2.imshow("Gesture Camera", frame)
            cv2.imshow("Wave Simulator (with legend & cross-section)", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'): sim.two_source=False
            if key == ord('d'): sim.two_source=True
            if key == ord('c'): sim.reset()

    except KeyboardInterrupt:
        pass
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
