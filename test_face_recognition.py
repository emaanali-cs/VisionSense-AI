import cv2
from app.face_recognition import recognize_faces

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = recognize_faces(frame)

    for face in faces:

        x1, y1, x2, y2 = face["bbox"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        text = f'{face["name"]} ({face["confidence"]:.2f})'

        cv2.putText(
            frame,
            text,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()