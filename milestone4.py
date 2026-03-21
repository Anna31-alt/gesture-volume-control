def get_detection_status(hand_area, hand_detected):
    if not hand_detected:
        return "No Detection", 0
    elif hand_area < 0.01:
        return "Poor Detection", 60
    else:
        return "Good Detection", 95