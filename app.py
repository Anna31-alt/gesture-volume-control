import cv2
import time
import streamlit as st

from milestone1 import init_camera, process_detection
from milestone2 import get_distance_and_gesture
from milestone3 import get_volume, draw_mapping_graph, draw_history
from milestone4 import get_detection_status

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Gesture Volume Control", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#FF4B4B;'>🎛 Gesture-Based Volume Control</h1>
<hr>
""", unsafe_allow_html=True)

# ---------------------------
# SESSION STATE
# ---------------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []

# ---------------------------
# BUTTONS
# ---------------------------
col1, col2, _ = st.columns([1,1,6])

with col1:
    if st.button("▶ Start"):
        st.session_state.run = True

with col2:
    if st.button("⏹ Stop"):
        st.session_state.run = False

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙ Control Panel")

st.sidebar.subheader("📊 Detection Status")
cam_status = st.sidebar.empty()
hands_status = st.sidebar.empty()
fps_status = st.sidebar.empty()
model_status = st.sidebar.empty()

st.sidebar.subheader("🎚 Detection Parameters")
detect_conf = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.75)
track_conf = st.sidebar.slider("Tracking Confidence", 0.5, 1.0, 0.80)

st.sidebar.subheader("📈 Detection Information")
landmark_text = st.sidebar.empty()
connection_text = st.sidebar.empty()
latency_text = st.sidebar.empty()

# ---------------------------
# MAIN LAYOUT
# ---------------------------
left, center, right = st.columns([1.2, 2.5, 1.3])

distance_text = left.empty()
distance_bar = left.progress(0)
accuracy_text = left.empty()

frame_placeholder = center.empty()

gesture_text = right.empty()
volume_text = right.empty()
volume_bar = right.progress(0)
graph_placeholder = right.empty()
history_graph = right.empty()

# ---------------------------
# RUN SYSTEM
# ---------------------------
if st.session_state.run:

    cap, hands = init_camera(detect_conf, track_conf)
    prev_time = time.time()

    while st.session_state.run:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results, hand, hand_count, landmark_count, connection_count, hand_area = process_detection(frame, hands)

        distance, gesture, volume = 0, "No Hand", 0

        if hand:
            distance, gesture = get_distance_and_gesture(frame, hand)
            volume = get_volume(distance)

        detection_status, accuracy = get_detection_status(hand_area, hand is not None)

        # FPS
        curr = time.time()
        fps = int(1 / (curr - prev_time)) if curr - prev_time > 0 else 0
        prev_time = curr

        latency = (time.time() - start) * 1000

        # CAMERA OVERLAY
        cv2.putText(frame, f"{gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"{detection_status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

        # SIDEBAR
        cam_status.success("Camera Active")
        hands_status.info(f"Hands Detected: {hand_count}")
        fps_status.warning(f"FPS: {fps}")
        model_status.success("Running")

        landmark_text.markdown(f"**Landmarks:** {landmark_count}")
        connection_text.markdown(f"**Connections:** {connection_count}")
        latency_text.markdown(f"**Latency:** {latency:.1f} ms")

        # LEFT PANEL
        distance_text.markdown(f"### Distance: {distance}")
        distance_bar.progress(min(distance / 200, 1.0))
        accuracy_text.success(f"Accuracy: {accuracy}%")

        # RIGHT PANEL
        gesture_text.success(gesture)
        volume_text.markdown(f"## 🔊 {volume}%")
        volume_bar.progress(volume / 100)

        # GRAPH
        st.session_state.volume_history.append(volume)
        if len(st.session_state.volume_history) > 50:
            st.session_state.volume_history.pop(0)

        graph_placeholder.pyplot(draw_mapping_graph(distance, volume))
        history_graph.pyplot(draw_history(st.session_state.volume_history))

        time.sleep(0.03)

    cap.release()