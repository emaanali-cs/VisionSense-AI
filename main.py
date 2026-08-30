from app.camera import open_camera, read_frame, close_camera
from app.detector import detect_person
from app.speech import speak
from app.utils import draw_box

import cv2
import time
import os
from datetime import datetime

# ==========================================
# Load DigiBoost Logo
# ==========================================

logo = cv2.imread(os.path.join("assets", "digiboost_logo.png"))

if logo is not None:
    logo = cv2.resize(logo, (48, 48))

# ==========================================
# Camera
# ==========================================

cap = open_camera()

# ==========================================
# Greeting Settings
# ==========================================

greeted = False
no_person_start = None

RESET_TIME = 7
WELCOME_DURATION = 3

# Show welcome banner when program starts
welcome_message = True
welcome_start = time.time()

# ==========================================
# FPS
# ==========================================

prev_time = time.time()

# ==========================================
# Main Loop
# ==========================================

while True:

    ret, frame = read_frame(cap)

    # --------------------------------------
    # Reconnect if stream disconnects
    # --------------------------------------

    if not ret:

        print("Camera disconnected... reconnecting")

        cap.release()

        time.sleep(1)

        cap = open_camera()

        continue

    person_found = False
    person_count = 0

    # ==========================================
    # Person Detection
    # ==========================================

    results = detect_person(frame)

    for result in results:

        for box in result.boxes:

            person_found = True
            person_count += 1

            confidence = float(box.conf[0])

            draw_box(frame, box, confidence)

    # ==========================================
    # Greeting Logic
    # ==========================================

    if person_found:

        no_person_start = None

        if not greeted:

            speak()

            greeted = True

            welcome_message = True

            welcome_start = time.time()

    else:

        if no_person_start is None:

            no_person_start = time.time()

        elif time.time() - no_person_start >= RESET_TIME:

            greeted = False

    # ==========================================
    # FPS
    # ==========================================

    current_time = time.time()

    fps = int(1 / max(current_time - prev_time, 0.001))

    prev_time = current_time

    # ==========================================
    # Dashboard Header
    # ==========================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], 80),
        (35, 35, 35),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )

    # ==========================================
    # DigiBoost Logo
    # ==========================================

    if logo is not None:

        frame[16:64, 15:63] = logo

    # ==========================================
    # Project Title
    # ==========================================

    cv2.putText(
        frame,
        "DigiBoost AI Vision",
        (78, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2
    )

    # ==========================================
    # ONLINE
    # ==========================================

    cv2.circle(
        frame,
        (82, 60),
        5,
        (0, 220, 120),
        -1
    )

    cv2.putText(
        frame,
        "ONLINE",
        (95, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    # ==========================================
    # People Count
    # ==========================================

    cv2.putText(
        frame,
        f"People: {person_count}",
        (205, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    # ==========================================
    # FPS
    # ==========================================

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (330, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    # ==========================================
    # Date Time
    # ==========================================

    date_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")

    text_size = cv2.getTextSize(
        date_time,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1
    )[0]

    x = frame.shape[1] - text_size[0] - 15

    cv2.putText(
        frame,
        date_time,
        (x, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1
    )

    # ==========================================
    # Welcome Banner
    # ==========================================

    if welcome_message:

        if time.time() - welcome_start < WELCOME_DURATION:

            cv2.rectangle(
                frame,
                (25, frame.shape[0] - 80),
                (460, frame.shape[0] - 35),
                (0, 120, 0),
                -1
            )

            cv2.putText(
                frame,
                "Welcome to DigiBoost Institute of Technology",
                (40, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        else:

            welcome_message = False

    # ==========================================
    # Footer
    # ==========================================

    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 25),
        (frame.shape[1], frame.shape[0]),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        frame,
        "Powered by Python | YOLOv8 | OpenCV",
        (15, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (frame.shape[1] - 130, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1
    )

    # ==========================================
    # Display
    # ==========================================

    cv2.imshow(
        "DigiBoost AI Person Detection System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# Cleanup
# ==========================================

close_camera(cap)
cv2.destroyAllWindows()