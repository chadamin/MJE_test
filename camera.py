import cv2

def get_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame