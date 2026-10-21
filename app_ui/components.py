import base64
import os
import hashlib, colorsys
import streamlit as st
import plotly.graph_objects as go
import config
from app_ui.theme import (SURFACE_2, SURFACE_3, SURFACE_4, BORDER, BORDER_SOFT, TEXT, TEXT_MUTED,
                           TEXT_FAINT, ACCENT, risk_color, VERSION)


# ---------------------------------------------------------------- detection color & grouping
_COLOR_CACHE = {}

def class_color_rgb(label):
    """Deterministic, visually distinct color per class label (0-255 RGB), so the same class
    always renders the same color across an image and across reruns."""
    if label not in _COLOR_CACHE:
        h = int(hashlib.md5(label.encode()).hexdigest(), 16)
        hue = (h % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
        _COLOR_CACHE[label] = (round(r * 255), round(g * 255), round(b * 255))
    return _COLOR_CACHE[label]

def class_color_hex(label):
    r, g, b = class_color_rgb(label)
    return f"#{r:02x}{g:02x}{b:02x}"

def class_color_bgr(label):
    r, g, b = class_color_rgb(label)
    return (b, g, r)

def contrast_text_bgr(rgb):
    return (0, 0, 0) if sum(rgb) > 380 else (255, 255, 255)

def _rects_overlap(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def draw_detections(img, detections, max_labels=40):
    """Draws color-coded, anti-aliased boxes with a filled, contrast-aware label background.
    In dense scenes (many overlapping boxes, e.g. heavy traffic), label backgrounds are nudged
    vertically to avoid stacking into an unreadable pile, and font scales down automatically
    with image size. Beyond max_labels detections, only the highest-confidence ones get a text
    label (boxes are still drawn for all) so the image stays legible instead of solid text."""
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    font_scale = max(0.35, min(0.5, w / 1000))
    thickness = 1 if font_scale < 0.45 else 1
    ranked = sorted(detections, key=lambda x: -x["confidence"])
    placed_label_rects = []

    for idx, d in enumerate(ranked):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        rgb = class_color_rgb(d["label"]); bgr = class_color_bgr(d["label"])
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2, lineType=cv2.LINE_AA)
        if idx >= max_labels:
            continue  # box only, no label — keeps very dense scenes legible
        text = f'{d["label"]} {d["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 4
        ty = max(th + pad * 2, y1)
        label_rect = (x1, ty - th - pad * 2, x1 + tw + pad * 2, ty)
        # nudge down repeatedly if this label's background collides with an already-placed one
        attempts = 0
        while any(_rects_overlap(label_rect, p) for p in placed_label_rects) and attempts < 6:
            shift = th + pad * 2
            label_rect = (label_rect[0], label_rect[1] + shift, label_rect[2], label_rect[3] + shift)
            attempts += 1
        lx1, ly1, lx2, ly2 = label_rect
        ly1, ly2 = max(0, ly1), min(h, ly2)
        placed_label_rects.append(label_rect)
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), bgr, -1, lineType=cv2.LINE_AA)
        cv2.putText(out, text, (lx1 + pad, ly2 - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    contrast_text_bgr(rgb), thickness, lineType=cv2.LINE_AA)
    return out

def grouped_detections(dets, label_key="label", conf_key="confidence"):
    """Collapses raw per-instance detections into one row per class — count + confidence range —
    the way a professional detection summary is presented, instead of one bar per instance."""
    groups = {}
    for d in dets:
        groups.setdefault(d[label_key], []).append(d[conf_key])
    out = [{"label": k, "count": len(v), "min_conf": min(v), "max_conf": max(v)} for k, v in groups.items()]
    return sorted(out, key=lambda g: -g["max_conf"])

def detection_group_row(g):
    color = class_color_hex(g["label"])
    rng = (f"{round(g['min_conf']*100)}–{round(g['max_conf']*100)}%" if g["count"] > 1
           else f"{round(g['max_conf']*100)}%")
    badge = f" × {g['count']}" if g["count"] > 1 else ""
    st.markdown(
        f'<div style="margin-bottom:0.6rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.85rem;'
        f'color:{TEXT_MUTED};margin-bottom:4px">'
        f'<span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
        f'background:{color};margin-right:7px"></span>{g["label"].replace("_"," ").title()}{badge}</span>'
        f'<span class="kp-mono" style="color:{TEXT}">{rng}</span></div>'
        f'<div class="kp-bar-track"><div class="kp-bar-fill" style="width:{round(g["max_conf"]*100)}%;'
        f'background:{color}"></div></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- navigation
def nav_group(title, items, active_page):
    st.sidebar.markdown(f'<div class="kp-nav-group">{title}</div>', unsafe_allow_html=True)
    for item in items:
        if item == active_page:
            st.sidebar.markdown(f'<div class="kp-nav-item kp-nav-active">{item}</div>', unsafe_allow_html=True)
        else:
            if st.sidebar.button(item, key=f"nav_btn_{item}"):
                st.session_state.active_page = item
                st.rerun()


def go_to(page_name):
    st.session_state.active_page = page_name
    st.rerun()


# ---------------------------------------------------------------- basic pieces
def metric_tile(label, value, delta=None, accent=ACCENT):
    """Stat tile. Long values (names, descriptions) automatically use a smaller font and wrap instead of overflowing."""
    d = f'<div class="kp-metric-delta">{delta}</div>' if delta else ""
    cls = "kp-metric-value" + (" kp-metric-value-sm" if len(str(value)) > 16 else "")
    st.markdown(
        f'<div class="kp-metric"><div class="kp-metric-icon" style="background:{accent}"></div>'
        f'<div class="kp-metric-label">{label.upper()}</div>'
        f'<div class="{cls}">{value}</div>{d}</div>', unsafe_allow_html=True)


def map_view(lats=None, lons=None, width_px=900, height_px=420):
    """Centre and zoom for a map. With points: fit them (Web-Mercator maths, clamped to zooms 8-15, single point = street
    level). Without points, or with unusable ones: Karachi. Maps therefore always open on the city - never on the world
    view that Plotly picks when it is not told where to look."""
    import math
    pts = []
    try:
        for la, lo in zip(list(lats or []), list(lons or [])):
            la, lo = float(la), float(lo)
            if math.isfinite(la) and math.isfinite(lo) and -85 < la < 85 and -180 <= lo <= 180:
                pts.append((la, lo))
    except (TypeError, ValueError):
        pts = []
    if not pts:
        return {"center": {"lat": config.KARACHI_CENTER[0], "lon": config.KARACHI_CENTER[1]}, "zoom": config.KARACHI_MAP_ZOOM}
    la0, la1 = min(p[0] for p in pts), max(p[0] for p in pts)
    lo0, lo1 = min(p[1] for p in pts), max(p[1] for p in pts)
    pad = 1.25                                            # breathing room around the outermost points
    span_lon = max((lo1 - lo0) * pad, 0.03)
    span_lat = max((la1 - la0) * pad, 0.03)
    z_lon = math.log2(360.0 * width_px / (256.0 * span_lon))
    # Mercator stretches vertical distance by 1/cos(lat): express the latitude span in longitude-degree equivalents
    z_lat = math.log2(360.0 * height_px * max(math.cos(math.radians((la0 + la1) / 2)), 0.1) / (256.0 * span_lat))
    return {"center": {"lat": (la0 + la1) / 2, "lon": (lo0 + lo1) / 2}, "zoom": round(max(8.0, min(15.0, z_lon, z_lat)), 2)}


def karachi_map(fig, height, lats=None, lons=None, width_px=900):
    """Applies the dark base map, a tight layout and a Karachi-fitted view to a Plotly map figure."""
    v = map_view(lats, lons, width_px, height)
    fig.update_layout(map_style="carto-darkmatter", map_center=v["center"], map_zoom=v["zoom"], height=height,
                      margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(font_color=TEXT, bgcolor="rgba(12,17,26,0.7)"), uirevision="krypta-map")
    return fig


def pill(text, color=None):
    style = f'border-color:{color};color:{color};' if color else ""
    st.markdown(f'<span class="kp-pill" style="{style}">{text}</span>', unsafe_allow_html=True)


def confidence_bar(label, value, color=ACCENT):
    pct = max(0, min(100, round(value * 100)))
    st.markdown(
        f'<div style="margin-bottom:0.55rem"><div style="display:flex;justify-content:space-between;'
        f'font-size:0.82rem;color:{TEXT_MUTED};margin-bottom:4px"><span>{label}</span>'
        f'<span class="kp-mono" style="color:{TEXT}">{pct}%</span></div>'
        f'<div class="kp-bar-track"><div class="kp-bar-fill" style="width:{pct}%;background:{color}"></div></div></div>',
        unsafe_allow_html=True)


def section_header(eyebrow, title, sub=None):
    sub_html = f'<div class="kp-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="kp-eyebrow">{eyebrow}</div><h1 style="margin-top:0">{title}</h1>{sub_html}',
                unsafe_allow_html=True)


def hero():
    """A restrained, single load-sequence hero — a city-grid field with pulsing observation
    nodes and slow connecting sweeps. Pure CSS/SVG, no external assets, respects reduced motion."""
    st.markdown(
        f"""
        <style>
        @keyframes kp-pulse {{ 0% {{ opacity:0.12; transform:scale(0.85); }} 50% {{ opacity:0.95; transform:scale(1.15); }}
                              100% {{ opacity:0.12; transform:scale(0.85); }} }}
        @keyframes kp-sweep {{ 0% {{ transform:translateX(-10%); opacity:0; }} 12% {{ opacity:0.5; }}
                              88% {{ opacity:0.5; }} 100% {{ transform:translateX(110%); opacity:0; }} }}
        @keyframes kp-rise {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
        @media (prefers-reduced-motion: reduce) {{ .kp-node, .kp-sweep {{ animation: none !important; }} }}
        .kp-hero {{
            position: relative; height: 200px; border-radius: 16px; overflow: hidden; margin-bottom: 1.3rem;
            background:
              radial-gradient(ellipse 70% 60% at 30% 20%, rgba(61,214,245,0.07), transparent 60%),
              repeating-linear-gradient(0deg, transparent, transparent 27px, {BORDER_SOFT} 28px),
              repeating-linear-gradient(90deg, transparent, transparent 27px, {BORDER_SOFT} 28px),
              {SURFACE_2};
            border: 1px solid {BORDER}; animation: kp-rise 0.6s ease-out;
        }}
        .kp-node {{ position:absolute; width:6px; height:6px; border-radius:50%; background:{ACCENT};
                   box-shadow: 0 0 10px 2px {ACCENT}; animation: kp-pulse 2.8s ease-in-out infinite; }}
        .kp-sweep {{ position:absolute; top:0; bottom:0; width:1px; background:
                    linear-gradient(180deg, transparent, {ACCENT}, transparent);
                    animation: kp-sweep 7s linear infinite; }}
        .kp-hero-label {{ position:absolute; bottom:14px; left:18px; font-family:'JetBrains Mono',monospace;
                          font-size:0.68rem; letter-spacing:0.08em; color:{TEXT_FAINT}; }}
        </style>
        <div class="kp-hero">
          <div class="kp-sweep" style="left:0%;"></div>
          <div class="kp-node" style="top:28%; left:18%; animation-delay:0s;"></div>
          <div class="kp-node" style="top:62%; left:34%; animation-delay:0.7s;"></div>
          <div class="kp-node" style="top:40%; left:55%; animation-delay:1.3s;"></div>
          <div class="kp-node" style="top:70%; left:72%; animation-delay:2.0s;"></div>
          <div class="kp-node" style="top:22%; left:83%; animation-delay:0.4s;"></div>
          <div class="kp-hero-label">LIVE OBSERVATION FIELD — SYNTHETIC PREVIEW</div>
        </div>
        """, unsafe_allow_html=True)


def radial_gauge(value, label, max_value=100, color=ACCENT, height=230):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": f" / {max_value}", "font": {"color": TEXT, "size": 32, "family": "Space Grotesk"}},
        gauge={"axis": {"range": [0, max_value], "tickcolor": TEXT_FAINT, "tickfont": {"color": TEXT_FAINT, "size": 10}},
               "bar": {"color": color, "thickness": 0.26}, "bgcolor": SURFACE_4,
               "borderwidth": 0, "steps": [{"range": [0, max_value], "color": SURFACE_4}]},
        title={"text": label, "font": {"color": TEXT_MUTED, "size": 12}}))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
    st.plotly_chart(fig, width="stretch")


def pipeline_diagram(stages, active_index=None):
    """A left-to-right methodology / processing pipeline. active_index highlights a live stage."""
    cols = st.columns(len(stages))
    for i, (c, s) in enumerate(zip(cols, stages)):
        active = active_index is not None and i <= active_index
        color = ACCENT if active else BORDER
        text_color = TEXT if active else TEXT_FAINT
        with c:
            st.markdown(
                f'<div style="text-align:center">'
                f'<div style="width:32px;height:32px;border-radius:50%;margin:0 auto 6px auto;'
                f'border:2px solid {color};display:flex;align-items:center;justify-content:center;'
                f'color:{color};font-weight:700;font-size:0.8rem;background:{SURFACE_2};'
                f'font-family:\'JetBrains Mono\',monospace">{i+1}</div>'
                f'<div style="font-size:0.68rem;color:{text_color};font-weight:600;letter-spacing:0.04em">{s}</div>'
                f'</div>', unsafe_allow_html=True)


def risk_contributors(res, weights=None):
    """Turns a risk_score() result into (label, points-out-of-100) pairs for a contributors chart."""
    w = weights or config.RISK_WEIGHTS
    total = sum(w.values()) or 1.0
    names = {"severity": "Severity", "exposure": "Traffic exposure", "recurrence": "Recurrence",
             "environment": "Environmental context", "confirmation": "Multi-source confirmation"}
    return [(names.get(k, k), round(100 * w.get(k, 0) * v / total, 1)) for k, v in res["components"].items()]


def incident_dossier(code, row, res):
    """Renders one incident as an intelligence dossier — shared by the Intelligence Map
    side panel and the Incidents page."""
    color = risk_color(res["level"])
    kind = str(row.get("type", "unknown")).replace("_", " ").title()
    st.markdown(
        f'<div class="kp-card" style="border-left:3px solid {color}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        f'<div><div class="kp-eyebrow" style="margin-bottom:2px">INCIDENT {code}</div>'
        f'<div style="font-size:1.2rem;font-weight:700" class="kp-display">{kind}</div></div>'
        f'<span class="kp-pill" style="border-color:{color};color:{color}">{res["level"].upper()} RISK</span>'
        f'</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_tile("Detection confidence", f'{round(row.get("severity", 0) * 100)}%')
    with c2: metric_tile("Evidence confidence", f'{round(row.get("evidence_confidence", 0) * 100)}%')
    with c3: metric_tile("Risk score", f'{res["score"]}', accent=color)

    st.markdown('<div class="kp-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;margin-bottom:0.5rem">WHY?</div>', unsafe_allow_html=True)
    if res["reasons"]:
        for r in res["reasons"]:
            st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.88rem;margin-bottom:3px">• {r.capitalize()}</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.88rem">No single dominant factor — risk is '
                    'evenly distributed across components.</div>', unsafe_allow_html=True)
    for n in res.get("uncertainty_notes", []):
        st.markdown(f'<div style="color:{risk_color("Moderate")};font-size:0.82rem;margin-top:6px">⚠ {n.capitalize()}</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Risk contributors — how was this calculated?"):
        for label, pts in risk_contributors(res):
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;'
                        f'color:{TEXT_MUTED};margin-bottom:3px"><span>{label}</span>'
                        f'<span class="kp-mono" style="color:{TEXT}">+{pts}</span></div>', unsafe_allow_html=True)
        st.caption("Each contributor is its configured weight × the component's value, rescaled to points "
                   "out of 100. Weights live in config/__init__.py and are not hard-coded per incident.")

    st.markdown('<div class="kp-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;margin-bottom:0.5rem">OBSERVATION HISTORY</div>', unsafe_allow_html=True)
    n_obs = int(row.get("n_observations", 1))
    st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.88rem">{n_obs} observation(s) on record.</div>',
                unsafe_allow_html=True)
    if row.get("date") is not None:
        st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.82rem;margin-top:4px">Last recorded: {row["date"]}</div>',
                    unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT_FAINT};font-size:0.78rem;margin-top:6px;font-style:italic">'
                f'{res["disclaimer"]}</div></div>', unsafe_allow_html=True)


def empty_state(title, message, cta_label=None, key=None):
    st.markdown(
        f'<div class="kp-card" style="text-align:center;padding:2.6rem 1.5rem">'
        f'<div style="font-size:1.05rem;font-weight:700;margin-bottom:0.45rem" class="kp-display">{title}</div>'
        f'<div style="color:{TEXT_MUTED};font-size:0.92rem;max-width:440px;margin:0 auto 1rem auto;line-height:1.5">{message}</div>'
        f'</div>', unsafe_allow_html=True)
    if cta_label:
        return st.button(cta_label, key=key or cta_label)
    return False


def error_state(user_message, exc: Exception = None):
    st.markdown(
        f'<div class="kp-card" style="border-left:3px solid {risk_color("High")}">'
        f'<div style="font-weight:700;margin-bottom:0.3rem" class="kp-display">ANALYSIS INTERRUPTED</div>'
        f'<div style="color:{TEXT_MUTED};font-size:0.9rem">{user_message}</div></div>',
        unsafe_allow_html=True)
    if exc is not None:
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


def footer():
    st.markdown('<div class="kp-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kp-footer">KRYPTA &nbsp;·&nbsp; Karachi Risk &amp; YOLO-driven Platform for Terrain Analysis &nbsp;·&nbsp; Independent research prototype &nbsp;·&nbsp; '
        f'Version {VERSION} &nbsp;·&nbsp; Not affiliated with any government or academic institution</div>',
        unsafe_allow_html=True)


def privacy_badge():
    st.markdown(
        f'<span class="kp-pill" style="border-color:{ACCENT};color:{ACCENT}">PRIVACY PROTECTED — faces blurred before analysis</span>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------- brand logo
_LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "krypta-logo-app.png")
_LOGO_CACHE = {}


def logo_html(width_px=210, path=None):
    """KRYPTA logo as an inline <img> (base64, so no static-file serving is needed). Returns "" if the file is
    missing or unreadable - callers then fall back to the text wordmark, so a missing logo never breaks a page."""
    path = path or _LOGO_PATH
    key = (path, int(width_px))
    if key not in _LOGO_CACHE:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            _LOGO_CACHE[key] = (f'<img src="data:image/png;base64,{data}" alt="KRYPTA - Karachi Risk &amp; YOLO-driven '
                                f'Platform for Terrain Analysis" style="width:{int(width_px)}px;max-width:100%;height:auto;'
                                f'display:block;margin:0.2rem 0 1.1rem 0"/>')
        except OSError:
            _LOGO_CACHE[key] = ""
    return _LOGO_CACHE[key]
