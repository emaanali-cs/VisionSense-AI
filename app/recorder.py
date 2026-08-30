import cv2
import os
from datetime import datetime


def create_video_writer(cap):

    os.makedirs("output", exist_ok=True)

    filename = datetime.now().strftime(
        "output/DigiBoost_%Y-%m-%d_%H-%M-%S.mp4"
    )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 120:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        filename,
        fourcc,
        fps,
        (width, height)
    )

    print("=" * 50)
    print("Recording Started")
    print(filename)
    print("=" * 50)

    return writer


def write_frame(writer, frame):

    if writer is not None:
        writer.write(frame)


def stop_recording(writer):

    if writer is not None:

        writer.release()

        print("=" * 50)
        print("Recording Saved")
        print("=" * 50)