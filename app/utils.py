import cv2

# ==========================================
# Colors
# ==========================================

BOX_COLOR = (0, 220, 120)      # Soft Green
TEXT_COLOR = (255, 255, 255)   # White
LABEL_BG = (45, 45, 45)        # Dark Gray

# ==========================================
# Draw Detection Box
# ==========================================

def draw_box(frame, box, confidence):

    # Bounding Box Coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # ------------------------------
    # Bounding Box
    # ------------------------------
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        BOX_COLOR,
        2
    )

    # ------------------------------
    # Label Text
    # ------------------------------
    label = f"Visitor | {confidence * 100:.0f}%"

    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        1
    )

    # ------------------------------
    # Label Background
    # ------------------------------
    cv2.rectangle(
        frame,
        (x1, y1 - 28),
        (x1 + text_width + 14, y1),
        LABEL_BG,
        -1
    )

    # ------------------------------
    # Label Text
    # ------------------------------
    cv2.putText(
        frame,
        label,
        (x1 + 7, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )