import cv2
import time

from app.face_recognition import recognize_faces

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================
# Variables
# ==========================

frame_count = 0
cached_faces = []

greeted = False
last_unknown_time = 0

prev_time = time.time()

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # ----------------------------
    # Face Recognition
    # ----------------------------

    if frame_count % 5 == 0:
        cached_faces = recognize_faces(frame)

    person_count = len(cached_faces)

    unknown_present = False

    for face in cached_faces:

        x1, y1, x2, y2 = face["bbox"]

        name = face["name"]
        confidence = face["confidence"]

        if name == "Unknown":
            color = (0, 0, 255)
            unknown_present = True
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    # ----------------------------
    # Greeting Logic
    # ----------------------------

    if unknown_present:

        if not greeted:

            print("Greeting Unknown Person...")

            # later:
            # speak()

            greeted = True

            last_unknown_time = time.time()

    else:

        if greeted:

            if time.time() - last_unknown_time > 5:

                greeted = False

    # ----------------------------
    # FPS
    # ----------------------------

    current_time = time.time()

    fps = int(1 / max(current_time - prev_time, 0.001))

    prev_time = current_time

    # ----------------------------
    # Header
    # ----------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 70),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "DigiBoost Face Recognition",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People : {person_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"FPS : {fps}",
        (230,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    status = "UNKNOWN DETECTED" if unknown_present else "SAFE"

    status_color = (0,0,255) if unknown_present else (0,255,0)

    cv2.putText(
        frame,
        status,
        (430,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )

    # ----------------------------
    # Footer
    # ----------------------------

    cv2.rectangle(
        frame,
        (0, frame.shape[0]-30),
        (frame.shape[1], frame.shape[0]),
        (40,40,40),
        -1
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (15, frame.shape[0]-8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )

    cv2.imshow(
        "DigiBoost Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()