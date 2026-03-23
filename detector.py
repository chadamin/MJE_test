from ultralytics import YOLO

# 모델 로드 (처음 실행 시 자동 다운로드)
model = YOLO("yolov8n.pt")  # 가장 가벼운 모델

def detect(frame):
    results = model(frame)[0]

    objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = model.names[cls]

        # confidence 필터
        if conf < 0.5:
            continue

        objects.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "confidence": conf
        })

    return objects