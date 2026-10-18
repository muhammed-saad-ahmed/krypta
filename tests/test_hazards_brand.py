"""Hazard-handling regressions (text + config consistency) and brand/rename checks."""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import config
from geospatial import satellite as sat
from scripts import osint_text as nlp


def _a(t):
    return nlp.analyze_text(t)


# ---------------------------------------------------------------- config consistency
def test_every_hazard_type_has_severity_and_satellite_applicability():
    types = set(nlp.HAZARD_CONCEPTS) | set(config.EDGE_HAZARD_CLASSES) | set(config.HAZARD_LABEL_MAP.values())
    for h in types:
        assert h in config.BASE_SEVERITY, f"{h} has no base severity"
        assert h in config.SAT_APPLICABILITY, f"{h} has no satellite applicability"
        assert 0.0 <= config.BASE_SEVERITY[h] <= 1.0 and 0.0 <= config.SAT_APPLICABILITY[h] <= 1.0


def test_label_map_values_are_known_hazards_and_dataset_yaml_matches():
    assert set(config.HAZARD_LABEL_MAP.values()) <= set(config.BASE_SEVERITY)
    text = open(os.path.join(ROOT, "data", "hazard_dataset.example.yaml"), encoding="utf-8").read()
    names = re.findall(r"^\s+\d+:\s*(\w+)\s*$", text, re.M)
    assert names == list(config.EDGE_HAZARD_CLASSES)


def test_only_area_scale_hazards_are_satellite_observable():
    assert config.SAT_APPLICABILITY["waterlogging"] > 0
    for small in ("pothole", "open_manhole"):
        assert config.SAT_APPLICABILITY[small] == 0.0


# ---------------------------------------------------------------- hazard text regressions (bugs found in review)
@pytest.mark.parametrize("text", ["nala overflow ho gaya", "pani jama ho gaya hai", "pothole ho gaya",
                                  "sadak toot gayi ho gaya", "gutter ubal gaya"])
def test_filler_words_never_mark_a_live_hazard_as_resolved(text):
    r = _a(text)
    assert r["hazard_type"] is not None and r["status"] == "active", (text, r)


@pytest.mark.parametrize("text", ["gaddha theek ho gaya", "pothole fixed yesterday thanks", "kaam mukammal ho gaya pothole ka",
                                  "there was a pothole but it is repaired now"])
def test_genuine_fix_reports_are_still_resolved(text):
    assert _a(text)["status"] == "resolved"


def test_resolved_claim_blocked_by_still_or_not_yet():
    r = _a("gutter ubal raha hai abhi tak theek nahi hua")
    assert r["hazard_type"] == "sewage_overflow" and r["status"] == "active"


@pytest.mark.parametrize("text", ["gutter ka dhakkan khula hai bohat khatarnak", "khula gutter bacchon ke liye khatarnak",
                                  "manhole cover stolen", "Open manhole near Nazimabad, bacha gir gaya"])
def test_open_manhole_variants(text):
    assert _a(text)["hazard_type"] == "open_manhole"


@pytest.mark.parametrize("text,expect", [("stagnant water breeding mosquitoes", "waterlogging"),
                                         ("pani bhar gaya underpass mein", "waterlogging"),
                                         ("sewage water overflowing on main road", "sewage_overflow"),
                                         ("kachre ka dher lag gaya hai", "garbage"),
                                         ("road collapsed near flyover", "road_damage")])
def test_more_hazard_phrases(text, expect):
    assert _a(text)["hazard_type"] == expect


@pytest.mark.parametrize("text", ["traffic jam on shahrah e faisal", "i love karachi weather", "no flooding today alhamdulillah",
                                  "pani nikal gaya, road clear"])
def test_non_hazard_posts_do_not_create_hazards(text):
    r = _a(text)
    assert r["hazard_type"] is None and r["hazard_strength"] == 0.0


def test_child_danger_raises_urgency():
    assert _a("open manhole, bacha gir gaya")["urgency"] > _a("open manhole")["urgency"]


# ---------------------------------------------------------------- brand
def test_brand_strings_are_krypta_everywhere():
    bad = re.compile(r"karachi[ _-]?pulse", re.I)
    allowed = {os.path.join("config", "__init__.py")}            # legacy-database fallback path only
    hits = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", ".venv")]
        for f in files:
            if f.endswith((".py", ".md", ".txt", ".example", ".yaml", ".json")):
                p = os.path.join(base, f)
                rel = os.path.relpath(p, ROOT)
                if rel in allowed or rel == os.path.join("tests", "test_hazards_brand.py"):
                    continue
                if bad.search(open(p, encoding="utf-8", errors="ignore").read()):
                    hits.append(rel)
    assert not hits, f"old brand name still present in {hits}"


def test_app_and_readme_show_full_krypta_name():
    full = "Karachi Risk"
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    app = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    assert "# KRYPTA" in readme and "YOLO-driven Platform for Terrain Analysis" in readme
    assert "KRYPTA" in app and full in app and os.path.exists(os.path.join(ROOT, "docs", "krypta-logo.png"))


def test_default_database_is_renamed_but_old_one_still_used(tmp_path, monkeypatch):
    import importlib
    monkeypatch.delenv("KP_DB_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    assert config.DB_PATH == "data/krypta.db"
    os.makedirs("data"); open("data/karachi_pulse.db", "w").close()
    importlib.reload(config)
    assert config.DB_PATH == "data/karachi_pulse.db"
    os.remove("data/karachi_pulse.db")
    importlib.reload(config)
    assert config.DB_PATH == "data/krypta.db"


# ---------------------------------------------------------------- logo in the app
def test_app_logo_is_transparent_png_and_embedded_in_sidebar():
    cv2 = pytest.importorskip("cv2")
    img = cv2.imread(os.path.join(ROOT, "docs", "krypta-logo-app.png"), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.ndim == 3 and img.shape[2] == 4          # has an alpha channel
    assert (img[..., 3] == 0).mean() > 0.3                                   # background really is transparent
    app = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    assert "logo_html(" in app and "st.sidebar" in app


def test_logo_html_inlines_image_and_falls_back_when_missing(tmp_path):
    pytest.importorskip("streamlit"); pytest.importorskip("plotly")
    from app_ui.components import logo_html
    html = logo_html(200)
    assert html.startswith("<img src=\"data:image/png;base64,") and "width:200px" in html and "KRYPTA" in html
    assert logo_html(200, path=str(tmp_path / "nope.png")) == ""             # missing file -> text wordmark fallback
