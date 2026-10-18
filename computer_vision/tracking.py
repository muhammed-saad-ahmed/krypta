def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1) * max(0, y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0

def track(frames_dets, thr=0.3):
    """Greedy IoU tracker: same object across frames -> ONE track (incident)."""
    tracks = []
    for f, dets in enumerate(frames_dets):
        for d in dets:
            best = max((t for t in tracks if t["label"] == d["label"] and t["last_frame"] >= f-3),
                       key=lambda t: iou(t["bbox"], d["bbox"]), default=None)
            if best and iou(best["bbox"], d["bbox"]) >= thr:
                best.update(bbox=d["bbox"], last_frame=f); best["confs"].append(d["confidence"])
            else:
                tracks.append({"label": d["label"], "bbox": d["bbox"], "first_frame": f,
                               "last_frame": f, "confs": [d["confidence"]]})
    return tracks
