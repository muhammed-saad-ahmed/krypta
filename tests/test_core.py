import sys, os, pytest; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, cv2
from risk_engine.scoring import risk_score, evidence_confidence, road_health
from geospatial.clustering import cluster_observations, valid_coord
from computer_vision.preprocessing import validate_upload, decode_image, InvalidInput, sample_frames
from computer_vision.tracking import track
from database.db import connect, insert_observation, insert_detection, fetch_observations, fetch_detections
from evaluation.metrics import evaluate_detections, evaluate_uploaded_set, evaluate_with_conditions
from scripts.generate_demo_data import make
from risk_engine.traffic import density_level
from computer_vision.detection import Detector, class_names, benchmark
from scripts.dataset_tools import (validate_images, detect_duplicates, class_distribution,
                                    random_split, geographic_split, dataset_summary)
from prediction.hotspots import recurring_hotspots
from experiments.tracker import log_experiment, list_experiments

def test_five_frames_one_incident():
    obs = [{"type": "pothole", "lat": 24.86 + i*1e-6, "lon": 67.0} for i in range(5)]
    assert len(cluster_observations(obs)) == 1
def test_different_types_not_merged():
    assert len(cluster_observations([{"type": "pothole", "lat": 24.86, "lon": 67.0}, {"type": "garbage", "lat": 24.86, "lon": 67.0}])) == 2
def test_track_merges_frames():
    d = {"label": "pothole", "confidence": .8, "bbox": [10, 10, 50, 50]}
    assert len(track([[d], [d], [d]])) == 1
def test_same_source_does_not_inflate():
    a = evidence_confidence([{"source_id": "u1", "confidence": .8}] * 5)
    assert a == evidence_confidence([{"source_id": "u1", "confidence": .8}])
    assert evidence_confidence([{"source_id": "u1", "confidence": .8}, {"source_id": "u2", "confidence": .8}]) > a
def test_risk_monotonic_and_reasons():
    base = {"type": "waterlogging", "severity": .8, "n_observations": 1, "evidence_confidence": .5, "traffic_exposure": .9}
    lo, hi = risk_score(base, rainfall_mm=0), risk_score(base, rainfall_mm=100)
    assert hi["score"] > lo["score"] and "high traffic exposure" in hi["reasons"]
    assert 0 <= lo["score"] <= 100
def test_missing_data_lowers_risk_confidence():
    r = risk_score({"type": "pothole", "severity": .5})
    assert r["risk_confidence"] != "High" and r["uncertainty_notes"]
def test_road_health_weights():
    c = {k: 100 for k in ["surface", "drainage", "cleanliness", "lighting", "traffic_exposure", "recurrence", "safety_evidence"]}
    assert road_health(c) == 100.0
def test_coords():
    assert valid_coord(24.86, 67.0, (24.7, 66.8, 25.6, 67.6)) and not valid_coord(91, 0) and not valid_coord(None, 1) and not valid_coord(10, 10, (24.7, 66.8, 25.6, 67.6))
def test_validation():
    with pytest.raises(InvalidInput): validate_upload("a.exe", 10, "image")
    with pytest.raises(InvalidInput): validate_upload("a.jpg", 10**9, "image")
    with pytest.raises(InvalidInput): decode_image(b"garbage")
    with pytest.raises(InvalidInput): sample_frames("/nonexistent.mp4")
def test_db(tmp_path):
    c = connect(str(tmp_path / "t.db"))
    c.execute("INSERT INTO incidents(code,type,is_synthetic) VALUES('X','pothole',1)")
    assert c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 1
def test_eval_unavailable_and_perfect():
    assert "unavailable" in evaluate_detections([[]], [])
    g = [{"label": "a", "bbox": [0, 0, 10, 10]}]
    assert evaluate_detections([[dict(g[0], confidence=.9)]], [g])["f1"] == 1.0
def test_demo_flagged_synthetic():
    assert make(10)["is_synthetic"].all()

def test_density_level_bands():
    assert density_level(0) == "Low"
    assert density_level(7) == "Moderate"
    assert density_level(15) == "High"
    assert density_level(999) == "Severe"

def test_class_names_and_benchmark_safe_when_unavailable():
    d = Detector()  # never available
    assert class_names(d) == []
    assert benchmark(d) is None

def test_evaluate_uploaded_set_no_overlap_returns_none():
    d = Detector()
    assert evaluate_uploaded_set(d, {"a.jpg": None}, {"b.jpg": []}) is None

def test_evaluate_uploaded_set_perfect_match():
    class FakeDetector:
        def available(self): return True
        def detect(self, img): return [{"label": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    gt = {"img1.jpg": [{"label": "car", "bbox": [0, 0, 10, 10]}]}
    res = evaluate_uploaded_set(FakeDetector(), {"img1.jpg": "fake_array"}, gt)
    assert res["overall"]["f1"] == 1.0
    assert res["n_images"] == 1
    assert res["per_image"][0]["predicted"] == 1

# ---------------------------------------------------------------- error analysis (condition-wise eval)
class _FakeDet:
    """Deterministic fake keyed by the actual filename passed to detect() — NOT by call order,
    so the outcome map can never be accidentally misaligned with evaluate_with_conditions'
    internal sorted() iteration over filenames. The caller must pass images={name: name}."""
    def __init__(self, outcomes):
        self.outcomes = outcomes  # {filename: 'hit' or 'miss'}
    def available(self): return True
    def detect(self, img_key):
        return [{"label": "pothole", "confidence": 0.9, "bbox": [0, 0, 10, 10]}] if self.outcomes[img_key] == "hit" else []

def test_evaluate_with_conditions_backward_compatible_plain_list():
    # old-format ground truth (plain list, no conditions) must still work, bucketed as 'unspecified'
    det = _FakeDet({"a.jpg": "hit"})
    gt = {"a.jpg": [{"label": "pothole", "bbox": [0, 0, 10, 10]}]}
    res = evaluate_with_conditions(det, {"a.jpg": "a.jpg"}, gt)
    assert res is not None
    assert "unspecified" in res["condition_breakdown"]
    assert res["condition_breakdown"]["unspecified"]["f1"] == 1.0

def test_evaluate_with_conditions_buckets_single_condition():
    det = _FakeDet({"rain1.jpg": "hit", "rain2.jpg": "miss", "clear1.jpg": "hit"})
    gt = {
        "rain1.jpg": {"annotations": [{"label": "pothole", "bbox": [0, 0, 10, 10]}], "conditions": ["rain"]},
        "rain2.jpg": {"annotations": [{"label": "pothole", "bbox": [0, 0, 10, 10]}], "conditions": ["rain"]},
        "clear1.jpg": {"annotations": [{"label": "pothole", "bbox": [0, 0, 10, 10]}], "conditions": []},
    }
    images = {k: k for k in gt}
    res = evaluate_with_conditions(det, images, gt)
    assert res["condition_breakdown"]["rain"]["n_images"] == 2
    assert res["condition_breakdown"]["rain"]["recall"] == 0.5  # rain1 hit, rain2 miss -> 1/2
    assert res["condition_breakdown"]["unspecified"]["recall"] == 1.0  # clear1 hit
    assert res["condition_breakdown"]["unspecified"]["n_images"] == 1

def test_evaluate_with_conditions_multi_tag_image_counts_in_both_buckets():
    det = _FakeDet({"bad.jpg": "miss"})
    gt = {"bad.jpg": {"annotations": [{"label": "pothole", "bbox": [0, 0, 10, 10]}], "conditions": ["rain", "blur"]}}
    res = evaluate_with_conditions(det, {"bad.jpg": "bad.jpg"}, gt)
    assert res["condition_breakdown"]["rain"]["n_images"] == 1
    assert res["condition_breakdown"]["blur"]["n_images"] == 1
    assert res["condition_breakdown"]["rain"]["recall"] == 0.0

def test_evaluate_with_conditions_no_overlap_returns_none():
    det = _FakeDet({})
    assert evaluate_with_conditions(det, {"x.jpg": "x.jpg"}, {"y.jpg": []}) is None

# ---------------------------------------------------------------- dataset tools
def test_dataset_summary_empty_dir_is_honest(tmp_path):
    s = dataset_summary(str(tmp_path))
    assert s["n_images"] == 0 and "note" in s

def test_validate_images_flags_corrupted(tmp_path):
    import cv2, numpy as np
    good = (np.random.rand(20, 20, 3) * 255).astype("uint8")
    cv2.imwrite(str(tmp_path / "good.jpg"), good)
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    r = validate_images(str(tmp_path))
    assert r["checked"] == 2 and r["valid"] == 1 and len(r["corrupted"]) == 1
    assert r["corrupted"][0].endswith("bad.jpg")

def test_detect_duplicates_finds_identical_copy(tmp_path):
    import cv2, numpy as np
    img = (np.random.rand(30, 30, 3) * 255).astype("uint8")
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    cv2.imwrite(str(tmp_path / "a_copy.jpg"), img)
    pairs = detect_duplicates(str(tmp_path))
    assert len(pairs) == 1 and pairs[0][2] == 0

def test_class_distribution_counts():
    anns = [{"label": "pothole"}, {"label": "pothole"}, {"label": "garbage"}]
    assert class_distribution(anns) == {"pothole": 2, "garbage": 1}

def test_random_split_covers_all_records():
    recs = [{"id": i} for i in range(20)]
    out = random_split(recs)
    assert sum(len(v) for v in out.values()) == 20

def test_geographic_split_requires_coords():
    with pytest.raises(ValueError):
        geographic_split([{"id": 1}])

def test_geographic_split_keeps_same_cell_together():
    # two points in the same ~0.01-degree cell must never land in different splits
    recs = [{"id": i, "lat": 24.8600 + i * 1e-6, "lon": 67.0100} for i in range(10)]
    out = geographic_split(recs)
    buckets = [k for k, v in out.items() if v]
    assert len(buckets) == 1  # all 10 points are in one grid cell -> must stay together

# ---------------------------------------------------------------- prediction
def test_recurring_hotspots_none_when_insufficient_data():
    import pandas as pd
    assert recurring_hotspots(pd.DataFrame()) is None
    assert recurring_hotspots(pd.DataFrame({"n_observations": [1, 2]})) is None

def test_recurring_hotspots_returns_qualifying_rows():
    import pandas as pd
    df = pd.DataFrame({"code": ["A", "B", "C"], "n_observations": [1, 5, 3]})
    out = recurring_hotspots(df, min_observations=3)
    assert list(out["code"]) == ["B", "C"]  # sorted descending by observation count

# ---------------------------------------------------------------- experiment tracking
def test_experiment_logging_roundtrip(tmp_path, monkeypatch):
    import experiments.tracker as tr
    monkeypatch.setattr(tr, "LOG_PATH", str(tmp_path / "log.json"))
    assert tr.list_experiments() == []
    tr.log_experiment("baseline-test", "yolov8n", "v0", {"conf": 0.35}, {"f1": 0.0})
    logs = tr.list_experiments()
    assert len(logs) == 1 and logs[0]["name"] == "baseline-test"

# ---------------------------------------------------------------- database persistence
def test_db_observation_and_detection_roundtrip(tmp_path):
    conn = connect(str(tmp_path / "t2.db"))
    obs_id = insert_observation(conn, source_id="test.jpg", lat=24.86, lon=67.01,
                                 evidence_type="image", model="yolo-baseline", model_version="yolov8s.pt",
                                 confidence=0.8, is_synthetic=False, processing_version="0.1.0")
    insert_detection(conn, obs_id, "car", 0.8, [0, 0, 10, 10])
    obs = fetch_observations(conn)
    dets = fetch_detections(conn)
    assert len(obs) == 1 and obs.iloc[0]["source_id"] == "test.jpg"
    assert len(dets) == 1 and dets.iloc[0]["label"] == "car"
