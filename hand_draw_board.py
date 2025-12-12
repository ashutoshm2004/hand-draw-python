import cv2
import mediapipe as mp
import numpy as np

# ============================
# CONSTANTS & CONFIG
# ============================

BRUSH_SIZE = 8
BRUSH_COLOR = (85, 40, 255)  # BGR equivalent of #ff2d55

ERASE_THRESHOLD = 20         # frames required to confirm erase
MISSING_TOLERANCE = 6        # tolerate missing frames

# ============================
# SETUP
# ============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# canvas (same size later)
canvas = None

# state
last_pos = None
cursor = np.array([np.nan, np.nan])
missing_frames = 0
erase_streak = 0


# ============================
# FINGER STATE DETECTION
# ============================

def finger_details(lm):
    """Replicates the JS function that detects which fingers are up."""
    def is_up(tip, pip):
        return lm[tip].y < lm[pip].y

    index_up = is_up(8, 6)
    middle_up = is_up(12, 10)
    ring_up = is_up(16, 14)
    pinky_up = is_up(20, 18)

    # thumb: horizontal separation
    thumb_up = abs(lm[4].x - lm[3].x) > 0.06

    count = sum([index_up, middle_up, ring_up, pinky_up, thumb_up])
    return index_up, middle_up, ring_up, pinky_up, thumb_up, count


# ============================
# MAIN LOOP
# ============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror for user comfort (same as JS version)
    frame = cv2.flip(frame, 1)

    # setup canvas after knowing frame size
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    overlay = frame.copy()

    if results.multi_hand_landmarks:
        missing_frames = 0
        lm = results.multi_hand_landmarks[0].landmark

        # ===== Finger states =====
        index_up, middle_up, ring_up, pinky_up, thumb_up, count = finger_details(lm)

        # ===== Index fingertip position =====
        h, w, _ = frame.shape
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)
        pos = np.array([x, y])

        # ===== Smoothing (same formula as JS version) =====
        if np.isnan(cursor[0]):
            cursor = pos.copy()
        else:
            cursor = cursor * 0.72 + pos * 0.28

        cx, cy = int(cursor[0]), int(cursor[1])

        # Draw cursor on overlay
        cv2.circle(overlay, (cx, cy), 9, (0, 0, 0), -1)

        # ===== Gesture Detection =====
        draw_condition = index_up and not middle_up
        move_condition = index_up and middle_up
        erase_visible = (count >= 4)

        # ===== ERASE LOGIC =====
        if erase_visible:
            erase_streak += 1
            if erase_streak >= ERASE_THRESHOLD:
                canvas[:] = 0
                last_pos = None
        else:
            erase_streak = 0

        # ===== DRAWING =====
        if draw_condition and erase_streak == 0:
            if last_pos is None:
                last_pos = (cx, cy)
            else:
                cv2.line(canvas, last_pos, (cx, cy), BRUSH_COLOR, BRUSH_SIZE, cv2.LINE_AA)
                last_pos = (cx, cy)

        elif move_condition:
            # Move only — keep last_pos so smooth resume
            pass

        else:
            # Not drawing — allow tolerance
            pass

    else:
        # ===== Landmarks Missing =====
        missing_frames += 1
        if missing_frames > MISSING_TOLERANCE:
            last_pos = None
            cursor[:] = np.nan

    # Combine drawing canvas with video
    output = cv2.addWeighted(overlay, 1.0, canvas, 1.0, 0)

    cv2.imshow("Hand Draw Board (Python Version)", output)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
