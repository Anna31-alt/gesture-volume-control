import numpy as np
import cv2

def get_distance_and_gesture(frame, hand):
    h, w, _ = frame.shape

    x1 = int(hand.landmark[4].x * w)
    y1 = int(hand.landmark[4].y * h)
    x2 = int(hand.landmark[8].x * w)
    y2 = int(hand.landmark[8].y * h)

    distance = int(np.hypot(x2 - x1, y2 - y1))

    if distance < 40:
        gesture = "Closed Hand"
    elif distance < 120:
        gesture = "Pinch"
    else:
        gesture = "Open Hand"

    cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 3)

    return distance, gesture