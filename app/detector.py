from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

def detect_person(frame):

    results = model(
        frame,
        classes=[0],
        conf=0.20,
        imgsz=416,
        verbose=False
    )

    return results