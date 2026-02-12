import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image

st.set_page_config(page_title="Hand Gesture Recognition System", layout="wide")

st.markdown("""
<style>
body {background-color: #f5f6fa;}
.status-box {
    padding: 15px;
    border-radius: 10px;
    background: #ffffff;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🖐 Hand Gesture Recognition System")

col1, col2, col3 = st.columns([1.2,3,1.2])

# Sidebar Controls
st.sidebar.header("Detection Parameters")
detect_conf = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.75)
track_conf = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.80)
max_hands = st.sidebar.slider("Max Number of Hands", 1, 4, 2)

start = st.sidebar.button("▶ Start")
stop = st.sidebar.button("⏹ Stop")
capture = st.sidebar.button("📸 Capture")

# Detection Status Panel
with col1:
    st.subheader("Detection Status")
    status_box = st.empty()

# Video Feed
with col2:
    frame_window = st.image([])

# Detection Info Panel
with col3:
    st.subheader("Detection Info")
    info_box = st.empty()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

if start:

    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=detect_conf,
        min_tracking_confidence=track_conf
    )

    prev_time = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        hand_count = 0
        landmarks = 0
        connections = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_count += 1
                landmarks += 21
                connections += 20

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time + 0.0001))
        prev_time = curr_time

        latency = int((1/fps)*1000) if fps > 0 else 0

        frame_window.image(frame, channels="BGR")

        status_box.markdown(f"""
        <div class="status-box">
        <b>Camera:</b> Active<br>
        <b>Hands:</b> {hand_count}<br>
        <b>FPS:</b> {fps}<br>
        <b>Model:</b> MediaPipe Hands
        </div>
        """, unsafe_allow_html=True)

        info_box.markdown(f"""
        <div class="status-box">
        <b>Landmarks:</b> {landmarks}<br>
        <b>Connections:</b> {connections}<br>
        <b>Resolution:</b> 640 x 480<br>
        <b>Latency:</b> {latency} ms
        </div>
        """, unsafe_allow_html=True)

        if capture:
            cv2.imwrite("captured_hand.png", frame)
            st.sidebar.success("Image Captured!")

        if stop:
            break

    cap.release()

st.sidebar.markdown("---")
st.sidebar.markdown("### Internship Project")
st.sidebar.markdown("Hand Gesture Recognition System")
