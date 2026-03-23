import numpy as np

prev_positions = {}

def process_data(data):
    objects = data["objects"]
    frame_size = data["frame_size"]

    num_objects = len(objects)

    # -------------------------
    # 1. 밀도
    # -------------------------
    frame_area = frame_size[0] * frame_size[1]
    density = num_objects / frame_area if frame_area > 0 else 0

    # -------------------------
    # 2. 평균 속도
    # -------------------------
    speeds = []

    for obj in objects:
        obj_id = obj["id"]
        cx, cy = obj["center"]

        if obj_id in prev_positions:
            px, py = prev_positions[obj_id]
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            speeds.append(dist)

        prev_positions[obj_id] = (cx, cy)

    avg_speed = np.mean(speeds) if speeds else 0

    # -------------------------
    # 3. 충돌 위험 (bbox 겹침)
    # -------------------------
    collision_risk = 0

    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            boxA = objects[i]["bbox"]
            boxB = objects[j]["bbox"]

            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if (xB - xA) > 0 and (yB - yA) > 0:
                collision_risk += 1

    return {
        "num_objects": num_objects,
        "density": density,
        "avg_speed": avg_speed,
        "collision_risk": collision_risk
    }