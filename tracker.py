import numpy as np

class ByteTrackerWrapper:
    def __init__(self):
        from yolox.tracker.byte_tracker import BYTETracker

        class Args:
            track_thresh = 0.5
            match_thresh = 0.8
            track_buffer = 30
            mot20 = False

        self.tracker = BYTETracker(Args(), frame_rate=30)

    def update(self, detections):
        """
        detections:
        [
          {
            "bbox": [x1,y1,x2,y2],
            "confidence": 0.9
          }
        ]
        """

        if len(detections) == 0:
            return []

        dets = []

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            score = obj.get("confidence", 1.0)

            dets.append([x1, y1, x2, y2, score])

        dets = np.array(dets)

        online_targets = self.tracker.update(dets, (640,480), (640,480))

        results = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            x, y, w, h = tlwh
            bbox = [int(x), int(y), int(x+w), int(y+h)]

            results.append({
                "id": tid,
                "bbox": bbox
            })

        return results