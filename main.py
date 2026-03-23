import cv2
import time

from camera import get_camera, read_frame
from detector import detect
from motion import MotionDetector
from fusion import fuse
from tracker import SimpleTracker
from processor import process_data

cap = get_camera()
tracker = SimpleTracker()
motion = MotionDetector()

while True:
    frame = read_frame(cap)
    if frame is None:
        break

    frame = cv2.resize(frame, (640,480))

    # -------------------------
    # 1. YOLO
    # -------------------------
    yolo_objs = detect(frame)

    # -------------------------
    # 2. Motion
    # -------------------------
    motion_boxes = motion.detect(frame)

    # -------------------------
    # 3. Fusion
    # -------------------------
    objects = fuse(yolo_objs, motion_boxes)

    # -------------------------
    # 4. Tracking
    # -------------------------
    tracks = tracker.update(objects)

    # -------------------------
    # 5. Processing
    # -------------------------
    data = {
        "objects": tracks,
        "frame_size": (frame.shape[1], frame.shape[0]),
        "timestamp": time.time()
    }

    kpi = process_data(data)

    # -------------------------
    # 6. 시각화
    # -------------------------
    for obj in tracks:
        x1, y1, x2, y2 = obj["bbox"]

        color = (0,255,0) if obj.get("source")=="yolo" else (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{obj['id']} {obj.get('label','')}",
                    (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.putText(frame, f"Count: {kpi['num_objects']}", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()