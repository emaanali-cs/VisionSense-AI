import cv2

def open_camera():

    ip = "192.168.1.45"

    url = f"rtsp://admin:123456@{ip}:554/unicast/c1/s0/live"

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to open camera")

    return cap


def read_frame(cap):
    return cap.read()


def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()