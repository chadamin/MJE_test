import cv2

class MotionDetector:
    def __init__(self):
        self.prev_gray = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return []

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:   # 노이즈 제거
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x+w, y+h])

        self.prev_gray = gray
        return boxes