import numpy as np
from utils import get_center, get_size, get_distance

prev_positions = {}

def process_data(data):
    objects = data["objects"]
    timestamp = data["timestamp"]

    num_objects = len(objects)

    speeds = []
    distances = []
    sizes = []

    centers = []

    # -------------------------
    # 1. center / size / speed
    # -------------------------
    for obj in objects:
        obj_id = obj["id"]
        bbox = obj["bbox"]

        center = get_center(bbox)
        w, h, area = get_size(bbox)

        obj["center"] = center
        obj["size"] = (w, h, area)

        centers.append(center)
        sizes.append(area)

        # -------------------------
        # speed (pixel/frame)
        # -------------------------
        if obj_id in prev_positions:
            prev_center = prev_positions[obj_id]
            dist = get_distance(center, prev_center)
            speeds.append(dist)

        prev_positions[obj_id] = center

    avg_speed = np.mean(speeds) if speeds else 0

    # -------------------------
    # 2. 객체 간 거리
    # -------------------------
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            d = get_distance(centers[i], centers[j])
            distances.append(d)

    avg_distance = np.mean(distances) if distances else 0

    # -------------------------
    # 3. 평균 크기
    # -------------------------
    avg_size = np.mean(sizes) if sizes else 0

    return {
        "num_objects": num_objects,
        "avg_speed": avg_speed,
        "avg_distance": avg_distance,
        "avg_size": avg_size
    }