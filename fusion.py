from utils import iou

def fuse(yolo_objs, motion_boxes):
    fused = []

    # -------------------------
    # 1. YOLO 객체 먼저 추가
    # -------------------------
    for obj in yolo_objs:
        fused.append(obj)

    # -------------------------
    # 2. Motion 중복 제거
    # -------------------------
    for mbox in motion_boxes:
        is_duplicate = False

        for obj in yolo_objs:
            if iou(mbox, obj["bbox"]) > 0.3:
                is_duplicate = True
                break

        # -------------------------
        # 3. YOLO에 없는 움직임만 추가
        # -------------------------
        if not is_duplicate:
            fused.append({
                "bbox": mbox,
                "label": "unknown",
                "confidence": 1.0,
                "source": "motion"
            })

    return fused