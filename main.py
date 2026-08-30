from app.camera import open_camera, read_frame, close_camera
from app.detector import detect_person
from app.face_recognition import recognize_faces
from app.speech import speak
from app.utils import draw_box
from app.recorder import (
    create_video_writer,
    write_frame,
    stop_recording,
)

import cv2
import os
import time
from datetime import datetime

# =====================================================
# DigiBoost Logo
# =====================================================

logo = cv2.imread(os.path.join("assets", "digiboost_logo.png"))

if logo is not None:
    logo = cv2.resize(logo, (48, 48))

# =====================================================
# Camera
# =====================================================

cap = open_camera()

if cap is None:
    print("Unable to connect to camera.")
    exit()

video_writer = create_video_writer(cap)

# =====================================================
# Greeting Settings
# =====================================================

greeted = False
no_person_start = None

RESET_TIME = 7
WELCOME_DURATION = 3

welcome_message = True
welcome_start = None

# =====================================================
# FPS
# =====================================================

prev_time = time.time()

# =====================================================
# Face Recognition Cache
# =====================================================

frame_count = 0
cached_faces = []

# =====================================================
# Main Loop
# =====================================================

while True:

    ret, frame = read_frame(cap)
    # Start welcome timer after first frame arrives
    if welcome_start is None:
        welcome_start = time.time()

    if not ret:

        print("Camera disconnected... reconnecting")

        stop_recording(video_writer)
        cap.release()

        time.sleep(1)

        cap = open_camera()

        if cap is None:
            continue

        video_writer = create_video_writer(cap)
        welcome_message = True
        welcome_start = None
        continue

    person_found = False
    person_count = 0

# =====================================================
# YOLO PERSON DETECTION
# =====================================================

    results = detect_person(frame)

    person_count = 0
    person_found = False

    for result in results:

        person_count += len(result.boxes)

    if person_count > 0:
        person_found = True

    # =====================================================
    # FACE RECOGNITION
    # =====================================================

    frame_count += 1
    unknown_present = False

    largest_box = None
    largest_area = 0

    # =====================================================
    # FIND LARGEST PERSON
    # =====================================================

    for result in results:

        for box in result.boxes:

            px1, py1, px2, py2 = map(
                int,
                box.xyxy[0]
            )

            area = (px2 - px1) * (py2 - py1)

            if area > largest_area:

                largest_area = area

                largest_box = (
                    px1,
                    py1,
                    px2,
                    py2
                )

    # =====================================================
    # RECOGNIZE ONLY LARGEST PERSON
    # =====================================================

    if largest_box is not None:

        px1, py1, px2, py2 = largest_box

        person_width = px2 - px1
        person_height = py2 - py1

        # =================================================
        # TIGHTER UPPER-BODY CROP
        # =================================================

        pad_x = int(person_width * 0.10)

        crop_x1 = max(
            0,
            px1 - pad_x
        )

        crop_x2 = min(
            frame.shape[1],
            px2 + pad_x
        )

        crop_y1 = py1

        crop_y2 = min(
            frame.shape[0],
            py1 + int(person_height * 0.60)
        )

        person_crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        if person_crop.size != 0:

            # =================================================
            # UPSCALE FACE REGION
            # =================================================

            UPSCALE = 1.5

            enlarged_crop = cv2.resize(
                person_crop,
                None,
                fx=UPSCALE,
                fy=UPSCALE,
                interpolation=cv2.INTER_CUBIC
            )

            # =================================================
            # RECOGNITION EVERY 10 FRAMES
            # =================================================

            if frame_count % 10 == 0:

                cached_faces = recognize_faces(
                    enlarged_crop
                )

            # =================================================
            # DRAW RECOGNITION RESULT
            # =================================================

            for face in cached_faces:

                fx1, fy1, fx2, fy2 = face["bbox"]

                # Convert coordinates back from
                # the 1.5x enlarged crop
                fx1 = int(fx1 / UPSCALE)
                fy1 = int(fy1 / UPSCALE)
                fx2 = int(fx2 / UPSCALE)
                fy2 = int(fy2 / UPSCALE)

                # Convert crop coordinates
                # back to original frame coordinates
                x1 = crop_x1 + fx1
                y1 = crop_y1 + fy1
                x2 = crop_x1 + fx2
                y2 = crop_y1 + fy2

                name = face["name"]
                confidence = face["confidence"]

                # =================================================
                # KNOWN PERSON
                # =================================================

                if name != "Unknown":

                    color = (0, 255, 0)

                    label = (
                        f"{name} "
                        f"({confidence * 100:.0f}%)"
                    )

                # =================================================
                # UNKNOWN VISITOR
                # =================================================

                else:

                    unknown_present = True

                    color = (0, 0, 255)

                    label = "Unknown Visitor"

                # =================================================
                # FACE BOX
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # =================================================
                # FACE LABEL
                # =================================================

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )


    

    # =====================================================
    # GREETING LOGIC
    # =====================================================

    if unknown_present:

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

    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()

    fps = int(1 / max(current_time - prev_time, 0.001))

    prev_time = current_time

    # =====================================================
    # Dashboard Header
    # =====================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], 80),
        (35, 35, 35),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
    )
        # =====================================================
    # DigiBoost Logo
    # =====================================================

    if logo is not None:

        frame[16:64, 15:63] = logo

    # =====================================================
    # Project Title
    # =====================================================

    cv2.putText(
        frame,
        "DigiBoost AI Vision",
        (78, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )

    # =====================================================
    # ONLINE Indicator
    # =====================================================

    cv2.circle(
        frame,
        (82, 60),
        5,
        (0, 220, 120),
        -1,
    )

    cv2.putText(
        frame,
        "ONLINE",
        (95, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    # =====================================================
    # Person Count
    # =====================================================

    cv2.putText(
        frame,
        f"People: {person_count}",
        (205, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (330, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    # =====================================================
    # Date & Time
    # =====================================================

    date_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")

    text_size = cv2.getTextSize(
        date_time,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1,
    )[0]

    x = frame.shape[1] - text_size[0] - 15

    cv2.putText(
        frame,
        date_time,
        (x, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )

    # =====================================================
    # Welcome Banner
    # =====================================================

    if welcome_message:

        if time.time() - welcome_start < WELCOME_DURATION:

            cv2.rectangle(
                frame,
                (25, frame.shape[0] - 80),
                (460, frame.shape[0] - 35),
                (0, 120, 0),
                -1,
            )

            cv2.putText(
                frame,
                "Welcome to DigiBoost Institute of Technology",
                (40, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        else:

            welcome_message = False

    # =====================================================
    # Footer
    # =====================================================

    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 25),
        (frame.shape[1], frame.shape[0]),
        (35, 35, 35),
        -1,
    )

    cv2.putText(
        frame,
        "Powered by Python | YOLOv8 | OpenCV",
        (15, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (frame.shape[1] - 130, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
    )

    # =====================================================
    # Save Recording
    # =====================================================

    write_frame(video_writer, frame)

    # =====================================================
    # Display Window
    # =====================================================

    cv2.imshow(
        "DigiBoost AI Person Detection System",
        frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================================
# Cleanup
# =====================================================

stop_recording(video_writer)

close_camera(cap)

cv2.destroyAllWindows()