from utils import get_center, iou

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = []

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            best_match = None
            best_iou = 0

            for track in self.tracks:
                i = iou(det["bbox"], track["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_match = track

            if best_iou > 0.3:
                best_match["bbox"] = det["bbox"]
                best_match["center"] = get_center(det["bbox"])
                updated_tracks.append(best_match)
            else:
                track = {
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "center": get_center(det["bbox"])
                }
                self.next_id += 1
                updated_tracks.append(track)

        self.tracks = updated_tracks
        return self.tracks