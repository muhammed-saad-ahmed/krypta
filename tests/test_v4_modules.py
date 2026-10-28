"""v2.2 modules: mAP evaluator, Karachi bounding-box filter, English + Urdu-script text hardening, ledger rollback."""
import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from database import db as kpdb
from evaluation.metrics import calculate_map, iou_matrix
from geospatial import gazetteer as gz
from risk_engine import fusion as fu
from scripts import osint_pipeline as op
from scripts import osint_text as nlp


# ====================================================================== 1. mAP evaluator
GT = {"a.jpg": [{"label": "pothole", "bbox": [10, 10, 50, 50]}, {"label": "open_manhole", "bbox": [100, 100, 140, 140]}],
      "b.jpg": [{"label": "pothole", "bbox": [0, 0, 20, 20]}]}


def _perfect(gt):
    return {k: [dict(d, confidence=1.0) for d in v] for k, v in gt.items()}


def test_map_perfect_predictions_score_one():
    r = calculate_map(GT, _perfect(GT))
    assert r["map_50"] == 1.0 and r["map_50_95"] == 1.0
    assert all(c["precision"] == 1.0 and c["recall"] == 1.0 and c["mean_iou"] == 1.0 for c in r["per_class"].values())


def test_map_duplicate_overlapping_boxes_count_as_false_positives():
    pred = {"a.jpg": [{"label": "pothole", "bbox": [10, 10, 50, 50], "confidence": .9},
                      {"label": "pothole", "bbox": [11, 11, 51, 51], "confidence": .8},      # same object again
                      {"label": "open manhole", "bbox": [100, 100, 130, 140], "confidence": .7}],
            "b.jpg": [{"label": "Pothole", "bbox": [0, 0, 20, 20], "confidence": .5}]}
    r = calculate_map(GT, pred)
    p, m = r["per_class"]["pothole"], r["per_class"]["open_manhole"]
    assert (p["tp"], p["fp"], p["fn"]) == (2, 1, 0) and p["recall"] == 1.0 and p["precision"] == pytest.approx(0.6667, abs=1e-3)
    assert p["ap50"] == pytest.approx(0.835, abs=1e-3)                    # hand-computed, 101-point interpolation
    assert m["mean_iou"] == pytest.approx(0.75) and m["ap50"] == 1.0 and m["ap50_95"] == pytest.approx(0.6)
    assert r["map_50"] == pytest.approx((0.835 + 1.0) / 2, abs=1e-3)


def test_map_missing_predictions_and_unknown_classes_and_voc_mode():
    assert calculate_map(GT, {})["map_50"] == 0.0
    r = calculate_map(GT, {"a.jpg": [{"label": "car", "bbox": [1, 1, 5, 5], "confidence": .6}]})
    assert r["spurious_classes"] == ["car"] and r["map_50"] == 0.0
    voc = calculate_map(GT, {"a.jpg": [{"label": "pothole", "bbox": [10, 10, 50, 50], "confidence": .9}]}, interpolation="voc")
    assert voc["per_class"]["pothole"]["ap50"] == pytest.approx(0.5)


def test_map_accepts_json_string_file_and_list_layout(tmp_path):
    flat = [dict(d, image=img, confidence=0.9) for img, v in GT.items() for d in v]
    f = tmp_path / "gt.json"
    f.write_text(json.dumps(GT))
    assert calculate_map(str(f), json.dumps(flat))["map_50"] == 1.0


def test_map_skips_and_logs_malformed_boxes(caplog):
    bad = {"a.jpg": [{"label": "pothole", "bbox": [float("nan"), 0, 1, 1], "confidence": .5},
                     {"label": "pothole", "bbox": [5, 5, 5, 9], "confidence": .5},          # zero width
                     {"bbox": [0, 0, 4, 4], "confidence": .5}]}                             # no label
    with caplog.at_level(logging.ERROR):
        r = calculate_map(GT, bad)
    assert r["invalid_boxes"] == 3 and r["map_50"] == 0.0 and "unusable entry" in caplog.text
    with pytest.raises(ValueError):
        calculate_map("{not json", GT)
    with pytest.raises(FileNotFoundError):
        calculate_map("/no/such/file.json", GT)


def test_iou_matrix_values():
    m = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert m[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


# ====================================================================== 2. Karachi bounding box
@pytest.mark.parametrize("lat,lon,ok", [(24.8607, 67.0100, True), (24.75, 66.85, True), (25.15, 67.35, True),
                                        (24.7499, 67.0, False), (25.1501, 67.0, False), (24.9, 66.8499, False),
                                        (24.9, 67.3501, False), (31.5204, 74.3587, False), (67.0, 24.9, False),
                                        (float("nan"), 67.0, False), (None, 67.0, False), ("abc", 1, False)])
def test_is_within_karachi_boundaries(lat, lon, ok):
    assert gz.is_within_karachi(lat, lon, log_rejection=False) is ok


def test_rejection_is_logged_and_counted(caplog):
    gz.reset_rejection_stats()
    with caplog.at_level(logging.ERROR):
        assert gz.is_within_karachi(31.52, 74.35, source="rss:lahore") is False
    assert "SPATIAL_REJECTION" in caplog.text and "rss:lahore" in caplog.text
    s = gz.rejection_stats()
    assert s["total"] == 1 and s["recent"][-1]["reason"] == "outside_karachi_bbox"
    with pytest.raises(gz.OutsideKarachiError):
        gz.require_within_karachi(0, 0, source="x")
    assert gz.require_within_karachi(24.9, 67.0) == (24.9, 67.0)


def test_report_location_rejection_rules():
    assert gz.report_location_rejection(lat=31.5, lon=74.3) == "source_coordinates_outside_karachi"
    assert gz.report_location_rejection(lat=24.9, lon=67.0) is None
    assert gz.report_location_rejection(text="pothole at 31.5204, 74.3587") == "text_coordinates_outside_karachi"
    assert gz.report_location_rejection(text="pothole at 24.9215, 67.0925") is None
    assert gz.report_location_rejection(text="pothole near Saddar, no coordinates") is None
    assert gz.report_location_rejection(text="version 3.14159, 2.71828 released") is None


def test_pipeline_drops_out_of_bounds_reports_before_storing(tmp_path):
    conn = kpdb.connect(str(tmp_path / "p.db"))
    pipe = op.OSINTPipeline(conn, fuse=False)
    stats = op.PipelineStats()
    posts = [op.RawPost("mock", "1", "pothole near Saddar", lat=24.86, lon=67.01, is_synthetic=True),
             op.RawPost("mock", "2", "pothole in Lahore", lat=31.55, lon=74.34, is_synthetic=True),
             op.RawPost("mock", "3", "huge pothole 40.7128, -74.0060 here", is_synthetic=True)]
    for p in posts:
        op.run_sync(pipe.process(p, stats))
    assert stats.rejected_out_of_bounds == 2 and stats.inserted == 1
    assert conn.execute("SELECT COUNT(*) FROM osint_posts").fetchone()[0] == 1


def test_fusion_inputs_are_limited_to_the_strict_box():
    assert config.KARACHI_ADMIN_BBOX == (24.75, 66.85, 25.15, 67.35)
    assert gz.KARACHI_LAT_RANGE == (24.75, 25.15) and gz.KARACHI_LON_RANGE == (66.85, 67.35)
    for place in gz.known_places():
        g = gz.geocode_text(place)
        assert g is None or gz.is_within_karachi(g["lat"], g["lon"], log_rejection=False), place


# ====================================================================== 3. English + Urdu-script text
@pytest.mark.parametrize("concept,en,ur", [(c, e, u) for c, e, u in nlp.ALIGNED_PAIRS])
def test_alignment_pairs_resolve_to_the_same_concept(concept, en, ur):
    assert nlp.analyze_text(f"There is a {en} here")["hazard_type"] == concept
    assert nlp.analyze_text(f"یہاں {ur} ہے")["hazard_type"] == concept


@pytest.mark.parametrize("text,concept", [("Flooded street in Clifton", "waterlogging"), ("cracked asphalt everywhere", "road_damage"),
                                          ("open manhole on the main road", "open_manhole"), ("sewage overflow at the corner", "sewage_overflow"),
                                          ("deep pothole near the bridge", "pothole"),
                                          ("سڑک خراب ہے", "road_damage"), ("کھلا مین ہول ہے", "open_manhole"),
                                          ("گٹر کا پانی بہہ رہا ہے", "sewage_overflow"), ("سیوریج کا مسئلہ ہے", "sewage_overflow")])
def test_user_specified_terms(text, concept):
    r = nlp.analyze_text(text)
    assert r["hazard_type"] == concept and r["supported"] and r["lang"] in ("en", "ur")


def test_urdu_normalisation_variants_match():
    for variant in ("مین ہول کھلا ہے", "مَیْن ہول کھلا ہے", "مین‌ہول کھلا ہے", "مين هول کھلا ہے"):
        assert nlp.analyze_text(variant)["hazard_type"] == "open_manhole", variant
    assert nlp.normalize_urdu("مَيْن") == "مین" and nlp.normalize_urdu("١٢٣") == "123"
    r = nlp.analyze_text("گڑھا ہے۔ سڑک ٹھیک ہو گئی ہے")                       # Urdu full stop splits sentences
    assert r["hazard_type"] == "pothole"


def test_roman_urdu_and_other_scripts_are_refused_not_scored():
    for t in ("pani jama hai, koi nahi sunta", "sadak mein gaddha hai aur bohat khatarnak hai", "gutter ubal raha hai bohat pareshan hain"):
        r = nlp.analyze_text(t)
        assert r["lang"] == "roman-ur" and r["supported"] is False and r["hazard_type"] is None
        assert r["keywords"] == ["~unsupported:roman-ur"] and "unsupported_language" in r["flags"]
    assert nlp.analyze_text("Привет, яма на дороге")["lang"] == "other"
    assert nlp.analyze_text("Gulshan ka road has a pothole")["hazard_type"] == "pothole"        # one marker word is not Roman Urdu
    assert nlp.detect_language("گڑھا ہے سڑک پر") == "ur" and nlp.detect_language("pothole شکریہ") in ("mixed", "en")


@pytest.mark.parametrize("text", ["Excellent work KMC, the road is full of potholes", "Great job! open manhole still there",
                                  "بہت شکریہ، سڑک میں گڑھا ہے", "شاباش، گٹر کا پانی سڑک پر ہے", "واہ واہ، کھلا مین ہول ہے"])
def test_rule_based_sarcasm_flags_praise_plus_hazard(text):
    r = nlp.analyze_text(text)
    assert r["sarcasm"] and r["sarcasm_rule"] == "praise+hazard" and r["needs_review"] and "sarcasm" in r["flags"]
    assert r["praise_terms"] and "~sarcasm" in r["keywords"]


@pytest.mark.parametrize("text", ["Great job fixing the pothole, thank you", "Excellent work, the open manhole has been covered",
                                  "Excellent work by the team", "بہت شکریہ", "There is no pothole here, great job", "The pothole near Saddar"])
def test_genuine_praise_or_plain_reports_are_not_flagged(text):
    r = nlp.analyze_text(text)
    assert not (r["sarcasm"] and r["sarcasm_rule"] == "praise+hazard"), (text, r)


def test_high_intensity_praise_detector_and_lexicon_hits():
    assert "excellent work" in nlp.detect_high_intensity_praise("Excellent work!!")
    assert "بہت شکریہ" in nlp.detect_high_intensity_praise("بہت شکریہ آپ کا")
    assert nlp.detect_high_intensity_praise("the road is a road") == []
    hits = nlp.lexicon_hits("open manhole and سڑک خراب")
    assert "open manhole" in hits["open_manhole"] and "سڑک خراب" in hits["road_damage"]


def test_mock_stream_is_english_and_urdu_script_only():
    stream = op.MockStream(seed=3)
    for hz, langs in op._TEMPLATES.items():
        assert set(langs) <= {"en", "ur"}, hz
    for _ in range(40):
        for raw in stream.one():
            assert nlp.detect_language(raw.text) in nlp.SUPPORTED_LANGUAGES, raw.text


# ====================================================================== 4. ledger snapshot / rollback / alert
@pytest.fixture
def db(tmp_path):
    return kpdb.connect(str(tmp_path / "k.db"))


def _tamper_payload(db, seq=2):
    db.executescript("DROP TRIGGER IF EXISTS trg_ledger_no_update;")
    db.execute("UPDATE evidence_ledger SET payload='{\"i\":99}' WHERE seq=?", (seq,))
    db.commit()


def test_snapshot_is_written_before_every_append_with_a_signed_manifest(db, tmp_path):
    kpdb.append_ledger(db, "t", {"i": 0})
    path = tmp_path / "krypta_backup.db"
    assert path.exists() and (tmp_path / "krypta_backup.db.manifest.json").exists()
    first = json.loads((tmp_path / "krypta_backup.db.manifest.json").read_text())
    assert first["ledger_head_seq"] == 0                                    # state BEFORE the first append
    kpdb.append_ledger(db, "t", {"i": 1})
    second = json.loads((tmp_path / "krypta_backup.db.manifest.json").read_text())
    assert second["ledger_head_seq"] == 1 and second["signature"] and kpdb.verify_backup(str(path))["ok"]


def test_tamper_at_append_rolls_back_quarantines_and_raises(db, tmp_path):
    for i in range(3):
        kpdb.append_ledger(db, "t", {"i": i})
    _tamper_payload(db)
    assert not kpdb.verify_ledger(db)["ok"]
    with pytest.raises(kpdb.LedgerTamperError) as ei:
        kpdb.append_ledger(db, "t", {"i": "new"})
    err = ei.value
    assert err.rolled_back and err.source == "append_guard" and err.restored_head[0] == 2
    assert err.quarantine_path and os.path.exists(err.quarantine_path)
    assert kpdb.verify_ledger(db)["ok"] and kpdb.ledger_head(db)[0] == 2        # back to the last verified snapshot
    triggers = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert "trg_ledger_no_update" in triggers                                    # append-only guards came back with the data
    ev = kpdb.last_tamper_event()
    assert ev and ev["rolled_back"] and not ev["acknowledged"]
    kpdb.acknowledge_tamper_event()
    assert kpdb.last_tamper_event() is None
    assert kpdb.append_ledger(db, "t", {"i": "after"})[0] == 3                  # the system keeps working


def test_full_audit_raises_on_chain_mismatch_and_can_be_inspected_without_raising(db):
    for i in range(3):
        kpdb.append_ledger(db, "t", {"i": i})
    assert fu.full_audit(db)["ok"]
    _tamper_payload(db)
    rep = fu.full_audit(db, raise_on_tamper=False)
    assert not rep["ok"] and not rep["chain"]["ok"] and kpdb.verify_ledger(db)["ok"] is False   # nothing was restored
    with pytest.raises(kpdb.LedgerTamperError) as ei:
        fu.full_audit(db)
    assert ei.value.rolled_back and ei.value.source == "full_audit" and kpdb.verify_ledger(db)["ok"]


def test_tampered_backup_is_never_restored(db, tmp_path):
    for i in range(3):
        kpdb.append_ledger(db, "t", {"i": i})
    with open(tmp_path / "krypta_backup.db", "ab") as fh:
        fh.write(b"corruption")
    _tamper_payload(db)
    with pytest.raises(kpdb.LedgerTamperError) as ei:
        fu.full_audit(db)
    assert not ei.value.rolled_back and "hash differs" in ei.value.rollback_error
    assert not kpdb.verify_ledger(db)["ok"]                                      # live data untouched, alert says NOT restored
    with pytest.raises(kpdb.BackupError):
        kpdb.rollback_to_backup(db)


def test_missing_backup_means_no_rollback_but_still_an_exception(db, tmp_path):
    for i in range(2):
        kpdb.append_ledger(db, "t", {"i": i})
    os.remove(tmp_path / "krypta_backup.db")
    _tamper_payload(db, seq=1)
    with pytest.raises(kpdb.LedgerTamperError) as ei:
        fu.full_audit(db)
    assert not ei.value.rolled_back and "does not exist" in ei.value.rollback_error


def test_missing_hmac_key_is_not_treated_as_tampering(db, monkeypatch):
    monkeypatch.setenv(config.LEDGER_KEY_ENV, "k1")
    kpdb.append_ledger(db, "t", {"i": 0})
    monkeypatch.delenv(config.LEDGER_KEY_ENV)
    with pytest.raises(kpdb.LedgerTamperError) as ei:
        fu.full_audit(db)
    assert not ei.value.rolled_back and "KP_LEDGER_KEY" in ei.value.rollback_error


def test_backup_can_be_disabled_and_in_memory_is_skipped(tmp_path, monkeypatch):
    mem = kpdb.connect(":memory:")
    kpdb.append_ledger(mem, "t", {"a": 1})                                       # no file -> no snapshot, no error
    monkeypatch.setattr(config, "LEDGER_BACKUP_ENABLED", False)
    d = kpdb.connect(str(tmp_path / "off.db"))
    kpdb.append_ledger(d, "t", {"a": 1})
    assert not (tmp_path / "krypta_backup.db").exists()


def test_alert_html_states_what_happened_and_escapes_input():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from app_ui.fusion_pages import tamper_alert_html
    html = tamper_alert_html({"message": "restored <b>x</b>", "rolled_back": True, "restored_head": (2, "h"),
                              "quarantine_path": "/tmp/krypta_corrupted_1.db", "at": "2026-10-28T00:00:00+00:00",
                              "report": {"chain": {"problems": [{"seq": 2, "problem": "entry hash mismatch <script>"}]}}})
    assert "ROLLED BACK" in html and "&lt;b&gt;" in html and "&lt;script&gt;" in html and "<script>" not in html
    assert "kp-alert" in html and "krypta_corrupted_1.db" in html
    bad = tamper_alert_html({"message": "m", "rolled_back": False, "rollback_error": "no backup", "report": {}})
    assert "NOT RESTORED" in bad and "no backup" in bad
