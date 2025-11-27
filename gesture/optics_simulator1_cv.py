# optics_simulator_cv.py
# Interactive Lens & Mirror Simulator using OpenCV
# Usage: python optics_simulator_cv.py
# Controls:
#  - Drag the object arrow (left) or the lens/mirror (vertical element) with mouse
#  - Trackbars: Mode (Lens/Mirror), Focal length px, Object height px, Show Rays, Reset
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]   # adjust number if file nested deeper
sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import math

# ---------- Configuration ----------
W, H = 1000, 600
axis_y = H // 2
lens_x_init = W * 3 // 5
obj_x_init = W // 5
obj_h_init = 100

WINDOW = 'Optics Simulator'

# state
state = {
    'mode': 0,            # 0 = lens, 1 = mirror
    'focal': 120,         # focal length in pixels (positive = converging / concave mirror)
    'obj_x': obj_x_init,
    'obj_h': obj_h_init,
    'lens_x': lens_x_init,
    'show_rays': 1,
    'dragging': None,     # 'object' or 'lens' or None
}

# ---------- Helpers ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def thin_lens_image(u, f):
    # u: object distance (positive if object is left of lens; we will pass signed u)
    # f: focal length (signed)
    # Solve 1/f = 1/v + 1/u -> v = 1 / (1/f - 1/u)
    # If denom ~ 0 => image at infinity
    eps = 1e-9
    denom = (1.0 / (f + eps) - 1.0 / (u + eps))
    if abs(denom) < 1e-12:
        return float('inf')
    v = 1.0 / denom
    return v

def mirror_image(u, f):
    # Mirror formula is same algebraic form (1/f = 1/u + 1/v),
    # but sign conventions matter. We'll assume:
    # - u positive when object is in front of mirror (left side for our layout)
    # - f positive for concave mirror, negative for convex.
    eps = 1e-9
    denom = (1.0 / (f + eps) - 1.0 / (u + eps))
    if abs(denom) < 1e-12:
        return float('inf')
    v = 1.0 / denom
    return v

def world_to_canvas_x(x):
    # state coordinates are already in canvas px; keep this wrapper for clarity
    return int(x)

# ---------- Drawing ----------
def draw_scene():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # background
    img[:] = (12, 18, 28)  # dark bluish
    # optical axis
    cv2.line(img, (0, axis_y), (W, axis_y), (180, 180, 180), 1, cv2.LINE_AA)

    lens_x = int(state['lens_x'])
    obj_x = int(state['obj_x'])
    obj_h = int(state['obj_h'])
    f = int(state['focal'])
    mode = state['mode']

    # draw lens or mirror
    if mode == 0:
        # lens: vertical rounded rectangle effect
        cv2.rectangle(img, (lens_x - 8, axis_y - 220), (lens_x + 8, axis_y + 220), (180, 220, 255), -1)
        cv2.rectangle(img, (lens_x - 8, axis_y - 220), (lens_x + 8, axis_y + 220), (90, 140, 180), 2)
        cv2.putText(img, 'Thin Lens', (lens_x - 42, axis_y - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,230,255), 1)
    else:
        # mirror: draw an arc (spherical mirror)
        radius = 220
        # arc center left of lens_x for a convex-looking shape? We'll draw semicircle centered at lens_x
        cv2.ellipse(img, (lens_x, axis_y), (radius, radius), 0, -90, 90, (180,220,255), 3, cv2.LINE_AA)
        cv2.putText(img, 'Mirror', (lens_x - 28, axis_y - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,230,255), 1)

    # draw object (arrow)
    obj_top = (obj_x, axis_y - obj_h)
    obj_bottom = (obj_x, axis_y)
    cv2.line(img, obj_bottom, obj_top, (255, 200, 80), 3, cv2.LINE_AA)
    cv2.rectangle(img, (obj_x - 6, axis_y - obj_h - 12), (obj_x + 6, axis_y - obj_h), (255,200,80), -1)
    cv2.putText(img, 'Object', (obj_x - 30, axis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,200,120), 1)

    # compute distances using sign conventions:
    # For lens: u = lens_x - obj_x (positive when object is left of lens)
    # For mirror: u = lens_x - obj_x (positive when object in front of mirror)
    u = lens_x - obj_x
    if u == 0:
        u = 1e-6

    # compute image distance v depending on mode
    if mode == 0:
        # lens
        v = thin_lens_image(u, state['focal'])
    else:
        # mirror
        v = mirror_image(u, state['focal'])

    # compute image coordinates and height
    if np.isfinite(v):
        image_x = lens_x + v
        # magnification m = -v/u (thin lens sign convention). For mirror same formula.
        if abs(u) < 1e-9:
            m = 0
        else:
            m = -v / u
        img_h = m * obj_h
    else:
        image_x = None
        m = None
        img_h = None

    # draw focal points on axis
    fpx = lens_x + state['focal']
    fnx = lens_x - state['focal']
    cv2.circle(img, (int(fpx), axis_y), 4, (120, 255, 200), -1)  # +f
    cv2.circle(img, (int(fnx), axis_y), 4, (255, 120, 200), -1)  # -f
    cv2.putText(img, f'f = {state["focal"]} px', (lens_x + 8, axis_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,230,255), 1)

    # draw image if finite
    if image_x is not None and np.isfinite(image_x):
        img_top = (int(image_x), int(axis_y - img_h))
        img_bottom = (int(image_x), axis_y)
        # choose color for real vs virtual: real (on opposite side) yellow, virtual cyan
        if (mode == 0 and v > 0) or (mode == 1 and v > 0):
            color = (255, 230, 120)  # real image
        else:
            color = (120, 235, 255)  # virtual
        cv2.line(img, img_bottom, img_top, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (int(image_x - 6), int(axis_y - img_h - 10)), (int(image_x + 6), int(axis_y - img_h)), color, -1)
        cv2.putText(img, 'Image', (int(image_x - 40), axis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # draw principal rays if enabled
    if state['show_rays'] == 1:
        # principal ray 1: parallel to axis from top of object -> through focal after lens (or reflect for mirror)
        y0 = axis_y - obj_h
        # from object to lens
        p0 = (obj_x, int(y0))
        p1 = (lens_x, int(y0))
        cv2.line(img, p0, p1, (255,255,255), 1, cv2.LINE_AA)

        if mode == 0:
            # refracted: through focal point on other side: line from lens to (lens_x + v, axis_y - img_h)
            if image_x is not None and np.isfinite(image_x):
                p2 = (int(image_x), int(axis_y - img_h))
                cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # go toward focal point
                p_f = (int(lens_x + state['focal']), axis_y)
                # line from lens passing through p_f
                dx = 800
                if state['focal'] != 0:
                    t = dx
                    p_far = (int(lens_x + t), int(y0 + (axis_y - p_f[1]) * (t / max(1, abs(state['focal'])))))
                    cv2.line(img, p1, p_far, (255,255,255), 1, cv2.LINE_AA)
        else:
            # mirror: ray reflects - a parallel ray reflects through focal point on same side
            # reflect direction symmetric around normal (approx): draw from lens to focal point mirrored
            p_f = (int(lens_x + state['focal']), axis_y)
            # reflect by drawing towards mirror focal
            dx = -800 if state['focal'] > 0 else 800
            p_far = (int(lens_x + dx), int(y0))
            cv2.line(img, p1, p_far, (255,255,255), 1, cv2.LINE_AA)

        # principal ray 2: through center of lens (undeviated approximately) -> straight line
        center = (lens_x, axis_y)
        cv2.line(img, (obj_x, int(y0)), (center[0] + 300, int(y0 + (center[1] - y0) * (300 / max(1, abs(lens_x - obj_x))))), (200,200,255), 1, cv2.LINE_AA)

        # principal ray 3: through focal point (object side) -> emerges parallel
        # for lens: ray from object top through focal point on object side up to lens, then parallel after lens
        focal_obj_side = (lens_x - state['focal'], axis_y)
        cv2.line(img, (obj_x, int(y0)), (int(focal_obj_side[0]), int(focal_obj_side[1])), (200,255,200), 1, cv2.LINE_AA)
        cv2.line(img, (int(focal_obj_side[0]), int(focal_obj_side[1])), (lens_x + 400, int(y0)), (200,255,200), 1, cv2.LINE_AA)

    # HUD text
    cv2.putText(img, f'u = {u:.1f} px', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)
    if np.isfinite(v):
        cv2.putText(img, f'v = {v:.1f} px', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)
        cv2.putText(img, f'm = {m:.2f}', (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)
        # show nature
        nature = "Real" if (v > 0) else "Virtual"
        inversion = "Inverted" if m < 0 else "Upright"
        cv2.putText(img, f'{nature}, {inversion}', (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)
    else:
        cv2.putText(img, f'v = infinity', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)

    cv2.putText(img, 'Controls: Drag object/lens, trackbars to change params, press ESC to exit', (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return img

# ---------- Callbacks ----------
def on_trackbar_change(val):
    # read trackbars into state
    state['mode'] = cv2.getTrackbarPos('Mode', WINDOW)  # 0 lens, 1 mirror
    state['focal'] = max(1, cv2.getTrackbarPos('Focal', WINDOW))
    state['obj_h'] = max(5, cv2.getTrackbarPos('ObjH', WINDOW))
    # show rays and obj_x/lens_x should be synced only through mouse not trackbar for simplicity
    state['show_rays'] = cv2.getTrackbarPos('Rays', WINDOW)

# mouse
def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # decide whether clicking object or lens
        obj_x = int(state['obj_x']); obj_h = int(state['obj_h']); lens_x = int(state['lens_x'])
        # is click near object arrow?
        if abs(x - obj_x) < 18 and abs(y - (axis_y - obj_h//2)) < 60:
            state['dragging'] = 'object'
        elif abs(x - lens_x) < 20 and abs(y - axis_y) < 220:
            state['dragging'] = 'lens'
    elif event == cv2.EVENT_MOUSEMOVE and state['dragging'] is not None:
        if state['dragging'] == 'object':
            # move object but clamp inside
            newx = clamp(x, 20, int(state['lens_x'] - 10))
            state['obj_x'] = newx
        elif state['dragging'] == 'lens':
            newx = clamp(x, 120, W - 120)
            state['lens_x'] = newx
    elif event == cv2.EVENT_LBUTTONUP:
        state['dragging'] = None

# reset helper (we do via trackbar 'Mode' change sometimes)
def reset_simulator():
    state['lens_x'] = lens_x_init
    state['obj_x'] = obj_x_init
    state['obj_h'] = obj_h_init
    state['focal'] = 120
    state['show_rays'] = 1

# ---------- Main UI ----------
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, W, H)
cv2.createTrackbar('Mode', WINDOW, state['mode'], 1, on_trackbar_change)    # 0 lens, 1 mirror
cv2.createTrackbar('Focal', WINDOW, state['focal'], 400, on_trackbar_change)  # px
cv2.createTrackbar('ObjH', WINDOW, state['obj_h'], 250, on_trackbar_change)
cv2.createTrackbar('Rays', WINDOW, state['show_rays'], 1, on_trackbar_change)
cv2.setMouseCallback(WINDOW, mouse_cb)

# main loop
while True:
    img = draw_scene()
    cv2.imshow(WINDOW, img)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        reset_simulator()
    # update state from trackbars occasionally
    on_trackbar_change(0)

cv2.destroyAllWindows()

