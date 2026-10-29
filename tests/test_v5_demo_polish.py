"""v2.3 demo polish: live->synthetic satellite failover, engine status badge, collapsed data-origin disclosure."""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import config
from database import db as kpdb
from geospatial import satellite as sat


class _Down(sat.SceneProvider):
    name = "planetary-computer"
    is_synthetic = False

    def __init__(self, error):
        self.error, self.calls = error, 0

    def search(self, bbox, start, end, max_cloud):
        self.calls += 1
        raise self.error


@pytest.fixture
def conn(tmp_path):
    return kpdb.connect(str(tmp_path / "s.db"))


def _failover(conn, error, cooldown=300.0):
    down = _Down(error)
    primary = sat.SatelliteVerifier(conn, down)
    return sat.FailoverSatelliteVerifier(conn, primary=primary, cooldown_s=cooldown), down


@pytest.mark.parametrize("error,reason", [(ConnectionError("Name or service not known"), "network unreachable"),
                                          (RuntimeError("401 Unauthorized: token expired"), "access token missing, expired or rejected"),
                                          (RuntimeError("429 Too Many Requests"), "API quota or rate limit reached"),
                                          (ImportError("No module named 'rasterio'"), "live satellite packages are not installed"),
                                          (RuntimeError("weird"), "provider error")])
def test_live_failure_is_answered_by_the_synthetic_layer_and_stays_labelled(conn, error, reason):
    fo, down = _failover(conn, error)
    ev = fo.verify(24.9215, 67.0925, "waterlogging", force=True)
    assert down.calls >= 1
    assert ev.is_synthetic and ev.provider == sat.SyntheticSatelliteProvider.name and ev.error is None
    assert ev.fallback_reason == reason and ev.fallback_from == "planetary-computer"
    assert fo.fallback_events and fo.fallback_events[-1]["reason"] == reason
    assert ev.verdict in (sat.CONFIRMED_WET, sat.NO_CHANGE, sat.INCONCLUSIVE)
    stored = kpdb.fetch_satellite_checks(conn)
    assert bool(stored.iloc[0]["is_synthetic"]) and stored.iloc[0]["provider"] == sat.SyntheticSatelliteProvider.name


def test_after_one_failure_the_live_provider_is_skipped_for_the_cooldown(conn):
    fo, down = _failover(conn, ConnectionError("offline"))
    fo.verify(24.92, 67.09, "waterlogging", force=True)
    calls = down.calls
    fo.verify(24.95, 67.05, "waterlogging", force=True)
    assert down.calls == calls and len(fo.fallback_events) == 2             # no second network timeout for the next report
    fo2, down2 = _failover(conn, ConnectionError("offline"), cooldown=0.0)
    fo2.verify(24.92, 67.09, "waterlogging", force=True)
    c2 = down2.calls
    fo2.verify(24.95, 67.05, "waterlogging", force=True)
    assert down2.calls > c2                                                 # cooldown over: live is tried again


def test_healthy_live_and_not_applicable_hazards_pass_through(conn):
    live = sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider())
    live.provider.is_synthetic = True
    fo = sat.FailoverSatelliteVerifier(conn, primary=live)
    ev = fo.verify(24.9215, 67.0925, "waterlogging", force=True)
    assert ev.fallback_reason is None and not fo.fallback_events
    na = fo.verify(24.9215, 67.0925, "pothole")
    assert na.verdict == sat.NOT_APPLICABLE and na.fallback_reason is None
    fo_down, down = _failover(conn, ConnectionError("x"))
    assert fo_down.verify(24.9, 67.0, "pothole").verdict == sat.NOT_APPLICABLE and down.calls == 0


def test_fallback_notice_wording_is_honest():
    msg = sat.FALLBACK_NOTICE.format(reason="network unreachable")
    assert "network unreachable" in msg and "Synthetic Mirror Layer" in msg and "simulated" in msg and "not real" in msg
    assert "High-Fidelity" not in msg and "cached" not in msg.lower()


def test_fallback_fields_do_not_change_the_result_digest(conn):
    fo, _ = _failover(conn, ConnectionError("x"))
    ev = fo.verify(24.9215, 67.0925, "waterlogging", force=True)
    before = ev.digest()
    ev.fallback_reason = "something else"
    assert ev.digest() == before and "fallback_reason" in ev.to_dict()
    row = kpdb.fetch_satellite_checks(conn).iloc[0].to_dict()
    assert sat.row_digest_matches(row)


def test_ui_wiring_uses_the_new_components_and_no_raw_warning_banners():
    app = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    pages = open(os.path.join(ROOT, "app_ui", "fusion_pages.py"), encoding="utf-8").read()
    comp = open(os.path.join(ROOT, "app_ui", "components.py"), encoding="utf-8").read()
    theme = open(os.path.join(ROOT, "app_ui", "theme.py"), encoding="utf-8").read()
    assert "st.warning(SYN_NOTICE" not in app and "st.warning(SYN_NOTE" not in pages
    assert "engine_status_html(" in app and "No vision model loaded" not in app and "No model is loaded" not in app
    assert 'with st.expander("System Diagnostic Integrity & Data Origin Disclosure")' in comp
    assert "Simulation &amp; Validation Layer" in comp and "KRYPTA Core Engine: Active" in comp
    assert "FailoverSatelliteVerifier(conn)" in pages and "render_fallback_notice" in pages
    for cls in (".kp-engine", ".kp-syn", ".kp-syn-dot"):
        assert cls in theme
