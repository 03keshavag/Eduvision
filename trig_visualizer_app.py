# trig_visualizer_app.py
"""
Interactive Trigonometric Graph Interface (Streamlit + Plotly)

- Enter an angle in degrees and pick a trig function.
- Graph updates in real time with a marker at the chosen angle.
- Supports sin, cos, tan, csc, sec, cot and handles discontinuities.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Trig Visualizer", layout="wide")

st.title("Interactive Trigonometric Graph Visualizer")
st.write("Enter an angle (degrees) and choose the trig function. The plot updates in real time.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    func = st.selectbox("Function", ["sin", "cos", "tan", "csc", "sec", "cot"])
    angle_deg = st.number_input("Angle (degrees)", value=30.0, step=1.0, format="%.3f")
    # slider for quick changes
    angle_slider = st.slider("Quick angle slider (deg)", min_value=-720, max_value=720, value=int(angle_deg))
    # If user moved slider, override number_input (keeps them in sync)
    if angle_slider != int(angle_deg):
        angle_deg = float(angle_slider)
        # update number_input on UI by rerender
        st.experimental_rerun()

    x_span = st.selectbox("X range (degrees)", ["-360 to 360", "-180 to 180", "-90 to 90", "0 to 360", "custom"])
    if x_span == "custom":
        left = st.number_input("Left°", value=-360, format="%d")
        right = st.number_input("Right°", value=360, format="%d")
        x_left, x_right = int(left), int(right)
    else:
        if x_span == "-360 to 360":
            x_left, x_right = -360, 360
        elif x_span == "-180 to 180":
            x_left, x_right = -180, 180
        elif x_span == "-90 to 90":
            x_left, x_right = -90, 90
        elif x_span == "0 to 360":
            x_left, x_right = 0, 360

    y_clip = st.slider("Y-axis clip (abs, >0)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    show_grid = st.checkbox("Show grid", value=True)
    st.markdown("---")
    st.write("Tips:")
    st.write("- For reciprocal functions (csc, sec, cot), values blow up near zeros — you'll see asymptotes.")
    st.write("- Use the slider for quick angle sweeping; number input for precise angles.")

# --- Helper math functions ---
def deg2rad(d):
    return d * np.pi / 180.0

def compute_trig(fn, x_rad):
    # returns y array, and a boolean mask for valid (not infinite/nan)
    if fn == "sin":
        y = np.sin(x_rad)
    elif fn == "cos":
        y = np.cos(x_rad)
    elif fn == "tan":
        y = np.tan(x_rad)
    elif fn == "csc":
        y = 1.0 / np.sin(x_rad)
    elif fn == "sec":
        y = 1.0 / np.cos(x_rad)
    elif fn == "cot":
        y = 1.0 / np.tan(x_rad)
    else:
        y = np.sin(x_rad)
    # mask out huge values (near asymptotes) to prevent plotting vertical spikes
    mask = np.isfinite(y)
    # also mask where absolute value too large
    mask = mask & (np.abs(y) < 1e6)
    return y, mask

# --- Build x axis in degrees and radians ---
x_degs = np.linspace(x_left, x_right, 3000)
x_rads = deg2rad(x_degs)

# --- Compute y and mask ---
y_vals, valid_mask = compute_trig(func, x_rads)

# We will break lines at discontinuities for better plotting:
# find indices where mask is False or jump is too large between adjacent valid points
break_mask = (~valid_mask)
# also break when difference between adjacent y's is large (indicates an asymptote)
diff = np.zeros_like(y_vals)
diff[1:] = np.abs(np.diff(y_vals))
big_jump = diff > (y_clip * 0.5)  # threshold to break lines
break_mask = break_mask | big_jump

# Prepare Plotly traces: we'll split x/y into continuous segments wherever break_mask True
segments_x = []
segments_y = []
current_x = []
current_y = []
for xi, yi, br in zip(x_degs, y_vals, break_mask):
    if br:
        if current_x:
            segments_x.append(np.array(current_x))
            segments_y.append(np.array(current_y))
            current_x = []
            current_y = []
        # skip including this point
    else:
        current_x.append(xi)
        current_y.append(yi)
# append any trailing segment
if current_x:
    segments_x.append(np.array(current_x))
    segments_y.append(np.array(current_y))

# Create Plotly figure
fig = go.Figure()
# add each continuous segment as a separate trace (so asymptotes are not connected)
for sx, sy in zip(segments_x, segments_y):
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(width=2), name=f"{func}(x)"))

# add a vertical marker at the chosen degree
x_sel = float(angle_deg)
y_sel, ok = compute_trig(func, np.array([deg2rad(x_sel)]))
y_point = y_sel[0] if ok[0] and np.isfinite(y_sel[0]) else None

# vertical line
fig.add_vline(x=x_sel, line_width=1, line_dash="dash", line_color="green", annotation_text=f"{x_sel:.3f}°", annotation_position="top left")

# point marker if finite and within clipping
if y_point is not None and np.abs(y_point) < 1e5:
    fig.add_trace(go.Scatter(x=[x_sel], y=[y_point], mode="markers+text", marker=dict(size=10, color="red"),
                             text=[f"{y_point:.4f}"], textposition="top center", name="value"))

# layout
fig.update_layout(title=f"{func}(x) — angle marker at {x_sel:.3f}°",
                  xaxis_title="Degrees",
                  yaxis_title=f"{func}(θ)",
                  showlegend=False,
                  template="plotly_white",
                  height=600)

# y axis limits: clip for readability (but keep autoscale if function is tame)
fig.update_yaxes(range=[-y_clip, y_clip], zeroline=True, showgrid=show_grid)
fig.update_xaxes(showgrid=show_grid)

# Add asymptote hints for tan/cot/sec/csc where denominator ~ 0
if func in ("tan", "sec"):
    # cos zeroes produce tan and sec asymptotes at cos(x)=0 -> x = 90 + k*180 degrees
    k_vals = np.arange(math.floor((x_left-90)/180)-1, math.ceil((x_right-90)/180)+1)
    for k in k_vals:
        xpos = 90 + 180*k
        if x_left - 1 <= xpos <= x_right + 1:
            fig.add_vline(x=xpos, line_width=1, line_dash="dot", line_color="gray", opacity=0.5)
if func in ("cot", "csc"):
    # sin zeroes produce csc and cot asymptotes at x = k*180 degrees
    k_vals = np.arange(math.floor(x_left/180)-1, math.ceil(x_right/180)+1)
    for k in k_vals:
        xpos = 180 * k
        if x_left - 1 <= xpos <= x_right + 1:
            fig.add_vline(x=xpos, line_width=1, line_dash="dot", line_color="gray", opacity=0.5)

# Draw the figure in the app
st.plotly_chart(fig, use_container_width=True)

# Numeric readout
st.markdown("---")
col1, col2, col3 = st.columns([1,1,1])
col1.metric("Angle (degrees)", f"{x_sel:.6f}")
if y_point is None or not np.isfinite(y_point) or abs(y_point) > 1e5:
    col2.metric("Value", "∞ (asymptote / undefined)")
else:
    col2.metric("Value", f"{y_point:.6f}")
col3.metric("Function", func)

# Optional: small interactive examples / presets
st.markdown("### Quick examples")
c1, c2, c3, c4 = st.columns(4)
if c1.button("sin 30°"):
    st.experimental_set_query_params(func="sin", angle=30)
    st.experimental_rerun()
if c2.button("cos 60°"):
    st.experimental_set_query_params(func="cos", angle=60); st.experimental_rerun()
if c3.button("tan 45°"):
    st.experimental_set_query_params(func="tan", angle=45); st.experimental_rerun()
if c4.button("csc 30°"):
    st.experimental_set_query_params(func="csc", angle=30); st.experimental_rerun()

st.markdown("### Notes")
st.write("- For reciprocal functions (csc/sec/cot) the function is undefined where denominator = 0; you will see asymptote lines.")
st.write("- Use the slider in the sidebar to sweep angles rapidly and watch the point move in real time.")
