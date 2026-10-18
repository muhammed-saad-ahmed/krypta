import time
import numpy as np
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1)*max(0, y2-y1); u = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u else 0.0
def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}
def evaluate_detections(preds, gts, iou_thr=0.5):
    """preds/gts: per-image lists of {label,bbox[,confidence]}. Greedy matching."""
    if not gts: return "Evaluation unavailable — trained model and labelled test set required."
    tp = fp = fn = 0
    for P, G in zip(preds, gts):
        used = set()
        for p in sorted(P, key=lambda d: -d.get("confidence", 0)):
            m = [(iou(p["bbox"], g["bbox"]), j) for j, g in enumerate(G) if j not in used and g["label"] == p["label"]]
            s, j = max(m, default=(0, -1))
            if s >= iou_thr: tp += 1; used.add(j)
            else: fp += 1
        fn += len(G) - len(used)
    return prf(tp, fp, fn)
def evaluate_uploaded_set(det, images, ground_truth, min_confidence=0.0):
    """Runs det on every image present in BOTH images and ground_truth (matched by filename) and
    computes REAL precision/recall/F1 at IoU>=0.5, plus measured inference time on this machine.
    images: {filename: BGR ndarray}. ground_truth: {filename: [{'label':str,'bbox':[x1,y1,x2,y2]}]}.
    Returns None if there is no filename overlap (nothing to evaluate)."""
    names = sorted(set(images) & set(ground_truth))
    if not names:
        return None
    preds, gts, per_image, times = [], [], [], []
    for n in names:
        t0 = time.time()
        raw = [d for d in det.detect(images[n]) if d["confidence"] >= min_confidence] if det.available() else []
        times.append((time.time() - t0) * 1000)
        preds.append(raw); gts.append(ground_truth[n])
        single = evaluate_detections([raw], [ground_truth[n]])
        row = {"file": n, "predicted": len(raw), "ground_truth": len(ground_truth[n])}
        row.update(single if isinstance(single, dict) else {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        per_image.append(row)
    overall = evaluate_detections(preds, gts)
    return {"overall": overall, "per_image": per_image,
            "mean_inference_ms": round(sum(times) / len(times), 1), "n_images": len(names)}


STANDARD_CONDITIONS = ["low_light", "night", "rain", "blur", "occlusion", "unusual_angle", "heavy_traffic"]


def evaluate_with_conditions(det, images, ground_truth, min_confidence=0.0):
    """Error-analysis evaluation: like evaluate_uploaded_set, but if a ground-truth entry carries
    a 'conditions' tag list — e.g. {"filename.jpg": {"annotations": [...], "conditions": ["rain","blur"]}}
    — results are ADDITIONALLY bucketed by condition, so weak spots (poor lighting, rain, occlusion,
    blur, unusual angle, night, heavy traffic) are visible as separate real numbers instead of being
    averaged away into one overall score. An image tagged with multiple conditions counts toward
    each bucket it's tagged with. Plain-list entries (no conditions) remain fully supported for
    backward compatibility and are bucketed as 'unspecified'.
    images: {filename: BGR ndarray}.
    ground_truth: {filename: [{'label','bbox'}] OR {'annotations': [...], 'conditions': [...]}}.
    Returns None if there is no filename overlap between images and ground_truth."""
    names = sorted(set(images) & set(ground_truth))
    if not names:
        return None

    normalized = {}
    for n in names:
        entry = ground_truth[n]
        if isinstance(entry, list):
            normalized[n] = (entry, [])
        else:
            normalized[n] = (entry.get("annotations", []), entry.get("conditions") or [])

    preds, gts, per_image, times = [], [], [], []
    condition_pairs = {}  # condition tag -> list of (pred, gt) for this image
    for n in names:
        anns, conditions = normalized[n]
        t0 = time.time()
        raw = [d for d in det.detect(images[n]) if d["confidence"] >= min_confidence] if det.available() else []
        times.append((time.time() - t0) * 1000)
        preds.append(raw); gts.append(anns)

        single = evaluate_detections([raw], [anns])
        row = {"file": n, "predicted": len(raw), "ground_truth": len(anns),
               "conditions": ", ".join(conditions) if conditions else "unspecified"}
        row.update(single if isinstance(single, dict) else {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        per_image.append(row)

        for tag in (conditions or ["unspecified"]):
            condition_pairs.setdefault(tag, []).append((raw, anns))

    overall = evaluate_detections(preds, gts)
    condition_breakdown = {}
    for tag, pairs in condition_pairs.items():
        p_list, g_list = [p for p, g in pairs], [g for p, g in pairs]
        metrics = evaluate_detections(p_list, g_list)
        metrics = metrics if isinstance(metrics, dict) else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        metrics["n_images"] = len(pairs)
        condition_breakdown[tag] = metrics

    return {"overall": overall, "per_image": per_image, "condition_breakdown": condition_breakdown,
            "mean_inference_ms": round(sum(times) / len(times), 1), "n_images": len(names)}


def mae_rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"mae": float(np.abs(y-yhat).mean()), "rmse": float(np.sqrt(((y-yhat)**2).mean()))}
