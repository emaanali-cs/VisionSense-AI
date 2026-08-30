import cv2

# =====================================================
# CAMERA MODE
# Change only this line:
#
# "laptop" = Laptop webcam
# "ip"      = DigiBoost IP Camera
# =====================================================

CAMERA_MODE = "ip"

# =====================================================


def open_camera():

    if CAMERA_MODE == "laptop":

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Failed to open laptop webcam")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Laptop webcam connected successfully.")

        return cap

    elif CAMERA_MODE == "ip":

        ip = "192.168.1.44"

        url = f"rtsp://admin:123456@{ip}:554/unicast/c1/s0/live"

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 20)

        if not cap.isOpened():
            print("Failed to open IP Camera")
            return None

        print("IP Camera connected successfully.")

        return cap

    else:

        print("Invalid CAMERA_MODE")

        return None


def read_frame(cap):
    for _ in range(2):
        cap.grab()

    ret, frame = cap.read()

    return ret, frame


def close_camera(cap):

    if cap is not None:

        cap.release()

    cv2.destroyAllWindows()