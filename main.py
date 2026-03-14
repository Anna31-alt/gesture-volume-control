import cv2
import time
import numpy as np
import streamlit as st
import mediapipe as mp
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Gesture Volume Control", layout="wide")

st.markdown(
"""
<h1 style='text-align:center;color:#1F77B4;'>🎛 Gesture-Based Volume Control System</h1>
<hr>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------------
# SESSION STATE VARIABLES
# ---------------------------------------------------------

if "run" not in st.session_state:
    st.session_state.run = False

if "capture" not in st.session_state:
    st.session_state.capture = False

if "volume_history" not in st.session_state:
    st.session_state.volume_history = []

# used for stability filtering
if "stable_volume" not in st.session_state:
    st.session_state.stable_volume = 0

if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()


# ---------------------------------------------------------
# BUTTON CONTROLS
# ---------------------------------------------------------
b1, b2, b3, _ = st.columns([1,1,1,6])

with b1:
    if st.button("▶ Start"):
        st.session_state.run = True

with b2:
    if st.button("⏹ Stop"):
        st.session_state.run = False

with b3:
    if st.button("📸 Capture"):
        st.session_state.capture = True


# ---------------------------------------------------------
# MEDIAPIPE INITIALIZATION
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
st.sidebar.header("⚙ Control Panel")

st.sidebar.subheader("📊 Detection Status")

cam_status = st.sidebar.empty()
hands_status = st.sidebar.empty()
fps_status = st.sidebar.empty()
model_status = st.sidebar.empty()

# detection parameters
st.sidebar.subheader("🎯 Detection Parameters")

detect_conf = st.sidebar.slider("Detection Confidence",0.5,1.0,0.75)
track_conf = st.sidebar.slider("Tracking Confidence",0.5,1.0,0.80)
max_hands = st.sidebar.slider("Maximum Hands",1,2,1)

st.sidebar.subheader("ℹ Detection Information")

landmark_text = st.sidebar.empty()
connection_text = st.sidebar.empty()
latency_text = st.sidebar.empty()

st.sidebar.markdown("**Resolution:** 640 × 480")


# ---------------------------------------------------------
# MAIN PAGE LAYOUT
# ---------------------------------------------------------
left, center, right = st.columns([1.2,2.5,1.3])


# ---------------------------------------------------------
# LEFT PANEL → DISTANCE INFO
# ---------------------------------------------------------
with left:

    st.subheader("📏 Distance Measurement")

    distance_text = st.empty()
    distance_bar = st.progress(0)
    distance_state = st.empty()
    accuracy_text = st.empty()


# ---------------------------------------------------------
# CENTER PANEL → CAMERA
# ---------------------------------------------------------
with center:

    st.subheader("📷 Live Camera Feed")

    frame_placeholder = st.empty()


# ---------------------------------------------------------
# RIGHT PANEL
# ---------------------------------------------------------
with right:

    st.subheader("✋ Gesture Status")
    gesture_text = st.empty()

    st.subheader("🔊 Volume Level")
    volume_text = st.empty()
    volume_bar = st.progress(0)

    st.subheader("📈 Distance → Volume Mapping")
    graph_placeholder = st.empty()

    st.subheader("📊 Volume History")
    history_graph = st.empty()


# ---------------------------------------------------------
# CAMERA INITIALIZATION
# ---------------------------------------------------------
cap = None
prev_time = time.time()


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
if st.session_state.run:

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    hands = mp_hands.Hands(
        min_detection_confidence=detect_conf,
        min_tracking_confidence=track_conf,
        max_num_hands=max_hands
    )

    while st.session_state.run:

        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        distance = 0
        volume = st.session_state.stable_volume
        hand_count = 0
        gesture = "No Hand"

        landmark_count = 0
        connection_count = 0

        detection_status = "No Detection"
        accuracy = 0


        # -------------------------------------------------
        # HAND DETECTION
        # -------------------------------------------------
        if results.multi_hand_landmarks:

            hand_count = len(results.multi_hand_landmarks)

            # Dual hand awareness:
            # detect both hands but control with FIRST hand only
            hand = results.multi_hand_landmarks[0]

            h, w, _ = frame.shape

            x1 = int(hand.landmark[4].x * w)
            y1 = int(hand.landmark[4].y * h)

            x2 = int(hand.landmark[8].x * w)
            y2 = int(hand.landmark[8].y * h)

            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)

            # distance between thumb and index
            distance = int(np.hypot(x2-x1,y2-y1))

            raw_volume = int(np.interp(distance,[20,200],[0,100]))
            raw_volume = np.clip(raw_volume,0,100)

            # -------------------------------------------------
            # GESTURE STABILITY FILTER
            # update only if stable for 0.5 sec
            # -------------------------------------------------
            if time.time() - st.session_state.last_update_time > 0.5:

                st.session_state.stable_volume = raw_volume
                st.session_state.last_update_time = time.time()

            volume = st.session_state.stable_volume

            landmark_count = 21
            connection_count = 20  # corrected value

            # gesture classification
            if distance < 40:
                gesture = "✊ Closed Hand"
            elif distance < 120:
                gesture = "🤏 Pinch"
            else:
                gesture = "🖐 Open Hand"

            # detection quality classification
            if distance < 30:
                detection_status = "Poor Detection"
                accuracy = 60
            else:
                detection_status = "Good Detection"
                accuracy = 95

            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

        else:

            detection_status = "No Detection"
            accuracy = 0


        # -------------------------------------------------
        # TEXT DISPLAY ON CAMERA
        # -------------------------------------------------
        color = (0,255,0)

        if detection_status == "Poor Detection":
            color = (0,165,255)

        if detection_status == "No Detection":
            color = (0,0,255)

        cv2.putText(
            frame,
            detection_status,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )


        # -------------------------------------------------
        # FPS CALCULATION
        # -------------------------------------------------
        current_time = time.time()
        delta = current_time - prev_time
        fps = int(1/delta) if delta>0 else 0
        prev_time = current_time

        # -------------------------------------------------
        # LATENCY
        # -------------------------------------------------
        latency = (time.time()-start_time)*1000


        # -------------------------------------------------
        # DISPLAY CAMERA
        # -------------------------------------------------
        frame_placeholder.image(frame,channels="BGR")


        # -------------------------------------------------
        # SIDEBAR STATUS
        # -------------------------------------------------
        cam_status.success("Camera Active")
        hands_status.info(f"Hands Detected: {hand_count}")
        fps_status.warning(f"FPS: {fps}")
        model_status.success("Model Running")

        landmark_text.markdown(f"**Landmarks:** {landmark_count}")
        connection_text.markdown(f"**Connections:** {connection_count}")
        latency_text.markdown(f"**Latency:** {latency:.1f} ms")


        # -------------------------------------------------
        # DISTANCE UI
        # -------------------------------------------------
        distance_text.markdown(f"**Distance:** {distance} px")

        distance_bar.progress(min(distance/200,1.0))

        accuracy_text.info(f"Detection Accuracy: {accuracy}%")

        if hand_count:
            distance_state.success("Hand Active")
        else:
            distance_state.error("No Hand")


        # -------------------------------------------------
        # GESTURE UI
        # -------------------------------------------------
        gesture_text.success(gesture)


        # -------------------------------------------------
        # VOLUME UI
        # -------------------------------------------------
        volume_text.markdown(f"### {volume}%")
        volume_bar.progress(volume/100)


        # -------------------------------------------------
        # STORE VOLUME HISTORY
        # -------------------------------------------------
        st.session_state.volume_history.append(volume)

        if len(st.session_state.volume_history) > 50:
            st.session_state.volume_history.pop(0)


        # -------------------------------------------------
        # DISTANCE → VOLUME GRAPH
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(4,2))

        x_line = np.linspace(20,200,50)
        y_line = np.interp(x_line,[20,200],[0,100])

        ax.plot(x_line,y_line)
        ax.scatter(distance,volume,s=120)

        ax.set_xlim(0,200)
        ax.set_ylim(0,100)

        ax.set_xlabel("Distance (px)")
        ax.set_ylabel("Volume (%)")

        ax.set_title("Distance → Volume Mapping")

        graph_placeholder.pyplot(fig)
        plt.close(fig)


        # -------------------------------------------------
        # VOLUME HISTORY GRAPH
        # -------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(4,2))

        ax2.plot(st.session_state.volume_history,color='purple')

        ax2.set_title("Volume History")
        ax2.set_ylabel("Volume %")

        history_graph.pyplot(fig2)
        plt.close(fig2)


        # -------------------------------------------------
        # SCREENSHOT FEATURE
        # -------------------------------------------------
        if st.session_state.capture:

            cv2.imwrite("gesture_capture.png",frame)

            st.session_state.capture = False


        time.sleep(0.03)


    cap.release()