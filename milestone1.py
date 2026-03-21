import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 🔒 Global lock for stable hand tracking
locked_hand_center = None

def init_camera(detect_conf, track_conf):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    hands = mp_hands.Hands(
        min_detection_confidence=detect_conf,
        min_tracking_confidence=track_conf,
        max_num_hands=2
    )

    return cap, hands


def get_hand_center(hand):
    x = [lm.x for lm in hand.landmark]
    y = [lm.y for lm in hand.landmark]
    return np.mean(x), np.mean(y)


def get_hand_area(hand):
    x = [lm.x for lm in hand.landmark]
    y = [lm.y for lm in hand.landmark]
    return (max(x) - min(x)) * (max(y) - min(y))


def process_detection(frame, hands):
    global locked_hand_center

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    selected_hand = None
    hand_count = 0
    landmark_count = 0
    connection_count = 0
    hand_area = 0

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)

        # Draw ALL hands
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # 🔒 FIRST FRAME LOCK
        if locked_hand_center is None:
            selected_hand = results.multi_hand_landmarks[0]
            locked_hand_center = get_hand_center(selected_hand)

        else:
            min_dist = float("inf")

            # 🔁 Track closest hand to locked one
            for hand in results.multi_hand_landmarks:
                cx, cy = get_hand_center(hand)
                dist = np.hypot(cx - locked_hand_center[0], cy - locked_hand_center[1])

                if dist < min_dist:
                    min_dist = dist
                    selected_hand = hand

            # Update lock
            if selected_hand:
                locked_hand_center = get_hand_center(selected_hand)

        if selected_hand:
            hand_area = get_hand_area(selected_hand)

        landmark_count = hand_count * 21
        connection_count = hand_count * 20

    else:
        locked_hand_center = None  # reset when no hand

    return results, selected_hand, hand_count, landmark_count, connection_count, hand_area