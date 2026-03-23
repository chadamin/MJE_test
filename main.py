import cv2
import time

from camera import get_camera, read_frame
from detector import detect
from tracker import SimpleTracker
from processor import process_data

cap = get_camera()
tracker = SimpleTracker()

while True:
    frame = read_frame(cap)
    if frame is None:
        break

    detections = detect(frame)
    tracks = tracker.update(detections)

    data = {
        "objects": tracks,
        "frame_size": (frame.shape[1], frame.shape[0]),
        "timestamp": time.time()
    }

    kpi = process_data(data)

    # -------------------------
    # 시각화
    # -------------------------
    for obj in tracks:
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["center"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {obj['id']}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(frame, f"Count: {kpi['num_objects']}", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.putText(frame, f"Speed: {kpi['avg_speed']:.2f}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()