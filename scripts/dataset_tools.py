"""
Dataset preparation utilities: corrupted-image validation, near-duplicate detection,
class-distribution stats, and train/val/test splitting (random OR geographic).

These operate on whatever is actually in data/raw — with no real dataset present yet,
running them reports an honest empty/zero result rather than a fabricated number.
"""
import os
import glob
import numpy as np


def list_images(directory):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(directory, "**", e), recursive=True)
    return sorted(files)


def validate_images(directory):
    """Tries to decode every image in directory. Returns counts + the list of files that
    failed to decode (corrupted / not actually an image)."""
    import cv2
    files = list_images(directory)
    corrupted = [f for f in files if cv2.imread(f) is None]
    return {"checked": len(files), "corrupted": corrupted, "valid": len(files) - len(corrupted)}


def _phash(path, hash_size=8):
    """Average-hash perceptual hash — no extra dependency beyond OpenCV/NumPy."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (hash_size, hash_size))
    avg = small.mean()
    return "".join("1" if v > avg else "0" for v in small.flatten())


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def detect_duplicates(directory, threshold=5):
    """Near-duplicate detection via perceptual hashing. Returns (file1, file2, distance)
    triples for every pair under the Hamming-distance threshold (lower = more similar)."""
    files = list_images(directory)
    hashes = {f: _phash(f) for f in files}
    hashes = {f: h for f, h in hashes.items() if h}
    items = list(hashes.items())
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = _hamming(items[i][1], items[j][1])
            if d <= threshold:
                pairs.append((items[i][0], items[j][0], d))
    return pairs


def class_distribution(annotations):
    """annotations: list of {'label': str, ...}. Returns a label -> count dict."""
    counts = {}
    for a in annotations:
        counts[a["label"]] = counts.get(a["label"], 0) + 1
    return counts


def dataset_summary(directory):
    """One-call honest summary: image count, corrupted count, resolution spread. Returns
    zeros and a note, not fabricated numbers, if the directory is empty."""
    import cv2
    files = list_images(directory)
    if not files:
        return {"n_images": 0, "corrupted": 0, "unique_resolutions": 0,
                "note": "No images found — add files to data/raw/ first."}
    resolutions, corrupted = [], 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            corrupted += 1
        else:
            resolutions.append((img.shape[1], img.shape[0]))
    return {"n_images": len(files), "corrupted": corrupted,
            "unique_resolutions": len(set(resolutions)), "resolutions": resolutions}


def random_split(records, ratios=(0.7, 0.15, 0.15), seed=42):
    """Ordinary random split by record, ignoring location — use geographic_split instead
    when records carry lat/lon, to avoid leaking nearby frames across train/val/test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n = len(records)
    n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    return {"train": [records[i] for i in tr], "val": [records[i] for i in va],
            "test": [records[i] for i in te]}


def geographic_split(records, ratios=(0.7, 0.15, 0.15), seed=42, cell_degrees=2):
    """Splits by whole geographic grid cell (rounded lat/lon), not by individual record, so
    nearby observations of the same physical location land in the same split — preventing the
    model from being tested on a location it effectively already saw in training.
    Raises ValueError if any record is missing lat/lon, rather than silently mis-splitting."""
    if not records:
        return {"train": [], "val": [], "test": []}
    missing = [r for r in records if r.get("lat") is None or r.get("lon") is None]
    if missing:
        raise ValueError(f"{len(missing)} record(s) missing lat/lon — geographic split needs coordinates for every record.")
    cell = lambda r: (round(r["lat"], cell_degrees), round(r["lon"], cell_degrees))
    cells = {}
    for r in records:
        cells.setdefault(cell(r), []).append(r)
    keys = list(cells.keys())
    np.random.default_rng(seed).shuffle(keys)
    n = len(keys)
    n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    train_k, val_k, test_k = keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:]
    out = {"train": [], "val": [], "test": []}
    for k in train_k: out["train"] += cells[k]
    for k in val_k: out["val"] += cells[k]
    for k in test_k: out["test"] += cells[k]
    return out
