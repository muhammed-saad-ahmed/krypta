import json
import logging
import math
import os
import time

import numpy as np

log = logging.getLogger(__name__)
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
    pred_json = {n: [{"label": d["label"], "bbox": d["bbox"], "confidence": d.get("confidence", 0.0)} for d in P]
                 for n, P in zip(names, preds)}
    gt_json = {n: [{"label": g["label"], "bbox": g["bbox"]} for g in G] for n, G in zip(names, gts)}
    return {"overall": overall, "per_image": per_image, "map": calculate_map(gt_json, pred_json),
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

    pred_json = {n: [{"label": d["label"], "bbox": d["bbox"], "confidence": d.get("confidence", 0.0)} for d in P]
                 for n, P in zip(names, preds)}
    gt_json = {n: [{"label": g["label"], "bbox": g["bbox"]} for g in G] for n, G in zip(names, gts)}
    return {"overall": overall, "per_image": per_image, "condition_breakdown": condition_breakdown,
            "map": calculate_map(gt_json, pred_json),
            "mean_inference_ms": round(sum(times) / len(times), 1), "n_images": len(names)}


def mae_rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"mae": float(np.abs(y-yhat).mean()), "rmse": float(np.sqrt(((y-yhat)**2).mean()))}



# ====================================================================== COCO-style mAP (pure NumPy)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))     # 0.50, 0.55, ... 0.95


def _load_json_like(source, what):
    """Accept a parsed object, a JSON string, or a path to a JSON file."""
    if isinstance(source, (dict, list)):
        return source
    if isinstance(source, (bytes, bytearray)):
        source = source.decode("utf-8")
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if not isinstance(source, str):
        raise TypeError(f"{what} must be a dict/list, a JSON string or a path to a JSON file, got {type(source).__name__}")
    stripped = source.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(source)
        except json.JSONDecodeError as e:
            raise ValueError(f"{what}: invalid JSON ({e})") from e
    if not os.path.isfile(source):
        raise FileNotFoundError(f"{what}: no such file: {source}")
    with open(source, encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"{what}: {source} is not valid JSON ({e})") from e


def _label_key(label):
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def _box_or_none(raw):
    """[x1, y1, x2, y2] -> float64 array with x1<x2, y1<y2, or None if it is unusable (NaN/inf/zero area)."""
    try:
        b = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if b.size != 4 or not np.all(np.isfinite(b)):
        return None
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _parse_records(data, what, need_conf):
    """Normalise to {image_id: [(label_key, label_original, box, confidence)]}; returns (records, n_invalid).
    Accepts {"img.jpg": [{label,bbox[,confidence]}]} or [{"image": "img.jpg", label, bbox[, confidence]}] or
    {"images": [...]} / {"annotations": [...]} wrappers around either list form."""
    if isinstance(data, dict) and ("images" in data or "annotations" in data or "detections" in data):
        wrapped = [data[k] for k in ("annotations", "detections", "images") if k in data and isinstance(data[k], list)]
        if not wrapped:
            raise ValueError(f"{what}: wrapper object has no list under 'annotations', 'detections' or 'images'")
        data = wrapped[0]
    flat = []
    if isinstance(data, dict):
        for image_id, items in data.items():
            if not isinstance(items, list):
                raise ValueError(f"{what}: value for image '{image_id}' must be a list of boxes")
            for it in items:
                flat.append((str(image_id), it))
    elif isinstance(data, list):
        for it in data:
            if not isinstance(it, dict) or not any(k in it for k in ("image", "image_id", "file", "filename")):
                raise ValueError(f"{what}: list form needs objects with an 'image' key")
            key = next(k for k in ("image", "image_id", "file", "filename") if k in it)
            flat.append((str(it[key]), it))
    else:
        raise ValueError(f"{what}: expected an object or a list, got {type(data).__name__}")
    records, invalid = {}, 0
    for image_id, it in flat:
        if not isinstance(it, dict):
            invalid += 1
            log.error("%s: image %s has a non-object entry (%r); skipped", what, image_id, it)
            continue
        label = it.get("label", it.get("class", it.get("category")))
        box = _box_or_none(it.get("bbox", it.get("box")))
        conf = it.get("confidence", it.get("score", 1.0 if not need_conf else None))
        if label is None or box is None or (need_conf and conf is None):
            invalid += 1
            log.error("%s: image %s has an unusable entry (label=%r bbox=%r confidence=%r); skipped",
                      what, image_id, label, it.get("bbox", it.get("box")), conf)
            continue
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            invalid += 1
            log.error("%s: image %s has a non-numeric confidence %r; skipped", what, image_id, conf)
            continue
        if not math.isfinite(conf):
            invalid += 1
            log.error("%s: image %s has a non-finite confidence; skipped", what, image_id)
            continue
        records.setdefault(image_id, []).append((_label_key(label), str(label).strip(), box, conf))
    return records, invalid


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU of two (N,4) / (M,4) xyxy arrays -> (N,M). Vectorised, no Python loops over pairs."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


def _average_precision(tp, n_gt, interpolation):
    """AP from a confidence-ordered TP/FP vector. 'coco101' = 101-point interpolation (COCO / Ultralytics);
    'voc' = exact area under the monotone precision envelope (Pascal VOC 2010+)."""
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    if interpolation == "voc":
        r = np.concatenate(([0.0], recall, [1.0]))
        p = np.concatenate(([envelope[0]], envelope, [0.0]))
        idx = np.nonzero(r[1:] != r[:-1])[0]
        return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))
    grid = np.linspace(0.0, 1.0, 101)
    pos = np.searchsorted(recall, grid, side="left")
    sampled = np.where(pos < recall.size, envelope[np.minimum(pos, recall.size - 1)], 0.0)
    return float(sampled.mean())


def _greedy_match(order_ious, thr):
    """order_ious: list (confidence-descending) of (image_id, iou_row over that image's GT boxes of this class).
    COCO rule: each prediction takes the highest-IoU GT that is still unmatched and has IoU >= thr; a second
    prediction on an already-claimed object is therefore a false positive (duplicate suppression)."""
    claimed = {}
    tp = np.zeros(len(order_ious), dtype=np.float64)
    matched_iou = np.zeros(len(order_ious), dtype=np.float64)
    for k, (image_id, row) in enumerate(order_ious):
        if row.size == 0:
            continue
        taken = claimed.setdefault(image_id, np.zeros(row.size, dtype=bool))
        candidates = np.where(taken, -1.0, row)
        j = int(np.argmax(candidates))
        if candidates[j] >= thr and candidates[j] > 0.0:
            taken[j] = True
            tp[k] = 1.0
            matched_iou[k] = row[j]
    return tp, matched_iou


def calculate_map(ground_truth_json, predicted_json, *, iou_thresholds=IOU_THRESHOLDS, conf_threshold=0.0,
                  interpolation="coco101"):
    """Object-detection evaluation with raw NumPy (no pycocotools / torch).

    ground_truth_json / predicted_json: a path to a JSON file, a JSON string, or the parsed object. Layouts:
        {"img1.jpg": [{"label": "pothole", "bbox": [x1, y1, x2, y2], "confidence": 0.91}, ...], ...}
        [{"image": "img1.jpg", "label": "pothole", "bbox": [...], "confidence": 0.91}, ...]
    Ground truth needs no confidence; predictions do (also accepted as "score"). Labels are matched
    case-insensitively with '_' and '-' treated as spaces, and reported with the ground-truth spelling.

    Returns a dict:
        map_50, map_50_95            mean AP over classes that have ground truth (NaN-free: 0.0 if there are none)
        per_class[label]             gt, predictions, tp, fp, fn, precision, recall, f1 (IoU 0.5, predictions with
                                     confidence >= conf_threshold), mean_iou (of matched pairs at IoU 0.5),
                                     ap50, ap50_95
        spurious_classes             predicted labels with no ground truth anywhere (counted as false positives,
                                     excluded from the mAP mean, as in COCO)
        invalid_boxes, n_images, n_gt, n_pred, iou_thresholds, interpolation
    Overlapping boxes: predictions are processed in confidence order and each ground-truth box can be claimed by only
    one prediction per threshold, so duplicate detections of one object are false positives.
    Malformed entries (NaN, zero-area, missing label) are skipped, logged with log.error and counted in invalid_boxes."""
    if interpolation not in ("coco101", "voc"):
        raise ValueError("interpolation must be 'coco101' or 'voc'")
    thresholds = tuple(float(x) for x in iou_thresholds)
    if not thresholds or any(not (0.0 < x <= 1.0) for x in thresholds):
        raise ValueError("iou_thresholds must be a non-empty sequence of values in (0, 1]")
    if 0.5 not in [round(x, 2) for x in thresholds]:
        raise ValueError("iou_thresholds must include 0.5 (needed for mAP@0.5 and precision/recall)")
    gt, bad_gt = _parse_records(_load_json_like(ground_truth_json, "ground_truth_json"), "ground_truth_json", False)
    pr, bad_pr = _parse_records(_load_json_like(predicted_json, "predicted_json"), "predicted_json", True)
    spelling = {}
    for items in gt.values():
        for key, original, _b, _c in items:
            spelling.setdefault(key, original)
    gt_classes = sorted(spelling)
    all_images = sorted(set(gt) | set(pr))
    per_class, ap50_list, ap_all_list = {}, [], []
    t50 = [i for i, x in enumerate(thresholds) if round(x, 2) == 0.5][0]
    for cls in gt_classes:
        gt_boxes = {img: np.array([b for k, _o, b, _c in gt.get(img, []) if k == cls]).reshape(-1, 4) for img in all_images}
        n_gt = int(sum(v.shape[0] for v in gt_boxes.values()))
        dets = [(c, img, b) for img in all_images for k, _o, b, c in pr.get(img, []) if k == cls]
        dets.sort(key=lambda d: -d[0])
        rows = [(img, iou_matrix(b[None, :], gt_boxes[img])[0]) for _c, img, b in dets]
        confs = np.array([d[0] for d in dets], dtype=np.float64)
        aps, stats50 = [], None
        for ti, thr in enumerate(thresholds):
            tp, miou = _greedy_match(rows, thr)
            aps.append(_average_precision(tp, n_gt, interpolation))
            if ti == t50:
                stats50 = (tp, miou)
        tp50, miou50 = stats50
        keep = confs >= conf_threshold
        tp_k = float(tp50[keep].sum())
        n_k = int(keep.sum())
        fp_k = n_k - tp_k
        fn_k = n_gt - tp_k
        precision = tp_k / n_k if n_k else 0.0
        recall = tp_k / n_gt if n_gt else 0.0
        matched = miou50[keep][tp50[keep] > 0]
        per_class[spelling[cls]] = {
            "gt": n_gt, "predictions": len(dets), "tp": int(tp_k), "fp": int(fp_k), "fn": int(fn_k),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "mean_iou": round(float(matched.mean()), 4) if matched.size else 0.0,
            "ap50": round(aps[t50], 4), "ap50_95": round(float(np.mean(aps)), 4)}
        ap50_list.append(aps[t50])
        ap_all_list.append(float(np.mean(aps)))
    spurious = sorted({o for items in pr.values() for k, o, _b, _c in items if k not in spelling})
    n_pred = sum(len(v) for v in pr.values())
    n_gt_total = sum(len(v) for v in gt.values())
    if not gt_classes:
        log.error("calculate_map: ground truth has no valid boxes; mAP is 0.0 by definition")
    for label in spurious:
        log.warning("calculate_map: predicted label '%s' has no ground truth; counted as false positives only", label)
    return {"map_50": round(float(np.mean(ap50_list)), 4) if ap50_list else 0.0,
            "map_50_95": round(float(np.mean(ap_all_list)), 4) if ap_all_list else 0.0,
            "per_class": per_class, "spurious_classes": spurious,
            "invalid_boxes": bad_gt + bad_pr, "n_images": len(all_images), "n_gt": n_gt_total, "n_pred": n_pred,
            "iou_thresholds": list(thresholds), "interpolation": interpolation}
