"""
KRYPTA design system — v2.
One place for color tokens, typography and the global CSS injected into every page.
Backend logic is untouched; this module only affects presentation.
"""
import streamlit as st

try:                                    # charts: dark, transparent template so plot areas are never light-grey blocks
    import plotly.graph_objects as _go
    import plotly.io as _pio
    _tpl = _go.layout.Template(_pio.templates["plotly_dark"])
    _tpl.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F5F7FA"),
                       xaxis=dict(gridcolor="#202938", zerolinecolor="#202938"),
                       yaxis=dict(gridcolor="#202938", zerolinecolor="#202938"))
    _pio.templates["krypta"] = _tpl
    _pio.templates.default = "krypta"
except Exception:                       # plotly missing or API change: charts keep their defaults, app still runs
    pass

BG = "#070A0F"
SURFACE_1 = "#0C111A"
SURFACE_2 = "#101722"
SURFACE_3 = "#141B27"
SURFACE_4 = "#182131"
BORDER = "#202938"
BORDER_SOFT = "#1A2230"
TEXT = "#F5F7FA"
TEXT_MUTED = "#8B95A7"
TEXT_FAINT = "#5C6579"
ACCENT = "#3DD6F5"
ACCENT_SOFT = "rgba(61,214,245,0.10)"
ACCENT_DIM = "#1B8AA3"

RISK_COLORS = {"Low": "#3AA76D", "Moderate": "#E8C547", "High": "#E8842C", "Critical": "#E14B4B",
               "Medium": "#E8842C"}

VERSION = "2.1.0"


def risk_color(level: str) -> str:
    return RISK_COLORS.get(level, TEXT_MUTED)


def inject_global_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
        .kp-display {{ font-family: 'Space Grotesk', 'Inter', sans-serif; }}
        .kp-mono {{ font-family: 'JetBrains Mono', monospace; }}

        .stApp {{ background: {BG}; color: {TEXT}; }}
        #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}
        .block-container {{ padding-top: 1.5rem; padding-bottom: 3.5rem; max-width: 1320px; }}
        ::selection {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
        *:focus-visible {{ outline: 2px solid {ACCENT} !important; outline-offset: 2px; }}

        /* ---------------- sidebar shell ---------------- */
        section[data-testid="stSidebar"] {{ background: {SURFACE_1}; border-right: 1px solid {BORDER}; }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; padding-left: 0.9rem; padding-right: 0.9rem; }}
        .kp-logo-row {{ display: flex; align-items: center; gap: 9px; margin-bottom: 0.15rem; }}
        .kp-logo-mark {{ width: 9px; height: 9px; border-radius: 2px; background: {ACCENT};
                         box-shadow: 0 0 9px 1px {ACCENT}; flex-shrink: 0; }}
        .kp-logo {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02rem;
                   letter-spacing: 0.01em; color: {TEXT}; }}
        .kp-logo-sub {{ color: {TEXT_FAINT}; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.09em;
                        margin: 0 0 1.5rem 18px; }}

        .kp-nav-group {{ color: {TEXT_FAINT}; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.09em;
                         margin: 1.05rem 0 0.3rem 0.7rem; }}
        .kp-nav-group:first-of-type {{ margin-top: 0.2rem; }}
        .kp-nav-item {{ font-size: 0.9rem; padding: 0.48rem 0.7rem; border-radius: 8px; margin-bottom: 1px;
                        border-left: 3px solid transparent; }}
        .kp-nav-active {{ background: {SURFACE_3}; color: {TEXT}; border-left: 3px solid {ACCENT}; font-weight: 600; }}

        section[data-testid="stSidebar"] .stButton {{ margin-bottom: 1px; }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: transparent !important; border: none !important; border-left: 3px solid transparent !important;
            color: {TEXT_MUTED} !important; font-weight: 500; text-align: left; justify-content: flex-start;
            padding: 0.48rem 0.7rem !important; border-radius: 8px !important; box-shadow: none !important;
            font-size: 0.9rem; transition: background 0.12s ease, color 0.12s ease;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: {SURFACE_2} !important; color: {TEXT} !important;
        }}
        section[data-testid="stSidebar"] .stButton > button p {{ text-align: left; font-size: 0.9rem; }}

        .kp-status {{ border-top: 1px solid {BORDER}; margin-top: 1.3rem; padding-top: 0.95rem; }}
        .kp-status-title {{ color: {TEXT_FAINT}; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.09em;
                            margin-bottom: 0.5rem; }}
        .kp-status-row {{ display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: {TEXT_MUTED};
                          margin-bottom: 5px; }}
        .kp-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}

        /* ---------------- headings ---------------- */
        h1 {{ font-family: 'Space Grotesk', sans-serif; color: {TEXT}; font-weight: 700; letter-spacing: -0.01em;
             font-size: 2.1rem; }}
        h2, h3 {{ color: {TEXT}; font-weight: 700; letter-spacing: -0.01em; }}
        p, span, div, label {{ color: {TEXT}; }}
        .kp-eyebrow {{ font-family: 'JetBrains Mono', monospace; color: {ACCENT}; font-size: 0.72rem; font-weight: 500;
                      letter-spacing: 0.1em; margin-bottom: 0.35rem; }}
        .kp-sub {{ color: {TEXT_MUTED}; font-size: 0.98rem; margin-top: -0.5rem; line-height: 1.5; max-width: 640px; }}

        /* ---------------- surfaces ---------------- */
        .kp-card {{ background: {SURFACE_2}; border: 1px solid {BORDER_SOFT}; border-radius: 14px;
                   padding: 1.15rem 1.35rem; margin-bottom: 0.9rem; }}
        .kp-card-tight {{ padding: 0.85rem 1.1rem; }}
        .kp-card-accent {{ border-color: {ACCENT_DIM}; }}

        .kp-metric {{ background: {SURFACE_2}; border: 1px solid {BORDER_SOFT}; border-radius: 14px;
                     padding: 1.05rem 1.2rem; height: 100%; position: relative; overflow: hidden; }}
        .kp-metric-icon {{ width: 8px; height: 8px; border-radius: 2px; margin-bottom: 0.55rem; }}
        .kp-metric-label {{ color: {TEXT_MUTED}; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; }}
        .kp-metric-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.05rem; font-weight: 700;
                            color: {TEXT}; line-height: 1.15; margin-top: 0.3rem; }}
        .kp-metric-delta {{ font-size: 0.76rem; color: {TEXT_FAINT}; margin-top: 0.25rem; }}

        .kp-pill {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.72rem;
                   font-weight: 600; border: 1px solid {BORDER}; color: {TEXT_MUTED}; letter-spacing: 0.02em; }}

        /* ---------------- inputs / controls ---------------- */
        .stButton > button, .stDownloadButton > button {{
            background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 9px;
            font-weight: 600; padding: 0.55rem 1.15rem; transition: all 0.15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
        button[kind="primary"] {{ background: {ACCENT} !important; color: #04141A !important; border: none !important; }}
        button[kind="primary"]:hover {{ filter: brightness(1.08); }}

        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
            background: {SURFACE_2} !important; color: {TEXT} !important; border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
        }}
        /* slider: only the thumb and the value label are touched; the track colours come from the theme's primaryColor */
        .stSlider [data-baseweb="slider"] [role="slider"] {{ background: {ACCENT} !important; box-shadow: none !important; }}
        .stSlider [data-testid="stSliderThumbValue"], .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {{ color: {TEXT} !important; background: transparent !important; }}

        /* multiselect / selectbox: closed state, chips and the text typed inside */
        .stMultiSelect div[data-baseweb="select"] > div {{
            background: {SURFACE_2} !important; color: {TEXT} !important; border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox input, .stMultiSelect input {{ color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important; }}
        .stSelectbox [data-baseweb="select"] span, .stMultiSelect [data-baseweb="select"] span {{ color: {TEXT}; }}
        span[data-baseweb="tag"] {{
            background: {ACCENT_SOFT} !important; border: 1px solid {ACCENT_DIM} !important; border-radius: 7px !important;
        }}
        span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg {{ color: {ACCENT} !important; fill: {ACCENT} !important; }}

        /* drop-down lists, calendars and menus open in a separate layer (a portal outside the widget), so they need
           their own rules - otherwise they inherit the browser's light colours (white list, white text) */
        div[data-baseweb="popover"], div[data-baseweb="popover"] > div {{
            background: {SURFACE_3} !important; color: {TEXT} !important; border-radius: 10px !important;
        }}
        div[data-baseweb="popover"] ul, ul[role="listbox"], div[data-baseweb="menu"] {{
            background: {SURFACE_3} !important; color: {TEXT} !important; border: 1px solid {BORDER};
        }}
        ul[role="listbox"] li, li[role="option"], div[data-baseweb="menu"] li {{
            background: transparent !important; color: {TEXT} !important;
        }}
        ul[role="listbox"] li *, li[role="option"] * {{ color: {TEXT} !important; }}
        ul[role="listbox"] li:hover, li[role="option"]:hover, li[role="option"][aria-selected="true"],
        li[role="option"][data-highlighted="true"], div[data-baseweb="menu"] li:hover {{
            background: {SURFACE_4} !important;
        }}
        ul[role="listbox"] li:hover *, li[role="option"]:hover *, li[role="option"][aria-selected="true"] * {{
            color: {ACCENT} !important;
        }}
        div[data-baseweb="calendar"], div[data-baseweb="calendar"] * {{ background-color: {SURFACE_3}; color: {TEXT}; }}

        /* other native widgets that default to light */
        .stTextArea textarea {{ background: {SURFACE_2} !important; color: {TEXT} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
        .stNumberInput button {{ background: {SURFACE_3} !important; color: {TEXT} !important; border-color: {BORDER} !important; }}
        input::placeholder, textarea::placeholder {{ color: {TEXT_FAINT} !important; opacity: 1; }}
        [data-testid="stTooltipContent"], div[data-baseweb="tooltip"] > div {{
            background: {SURFACE_4} !important; color: {TEXT} !important; border: 1px solid {BORDER};
        }}
        [data-testid="stCode"] pre, [data-testid="stCode"] code, .stCodeBlock pre {{
            background: {SURFACE_2} !important; color: {TEXT} !important;
        }}
        [data-testid="stJson"], [data-testid="stJson"] * {{ background-color: {SURFACE_2}; }}
        [data-baseweb="tab-list"] button {{ color: {TEXT_MUTED}; }}
        [data-baseweb="tab-list"] button[aria-selected="true"] {{ color: {ACCENT}; }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {TEXT}; }}
        .stRadio > label, .stCheckbox > label {{ color: {TEXT_MUTED}; }}
        div[role="radiogroup"] label[data-testid="stWidgetLabel"] {{ color: {TEXT}; }}

        [data-testid="stDataFrame"] {{ border: 1px solid {BORDER_SOFT}; border-radius: 10px; overflow: hidden; }}
        hr {{ border-color: {BORDER}; }}

        [data-testid="stFileUploaderDropzone"] {{
            background: {SURFACE_2}; border: 1.5px dashed {BORDER}; border-radius: 16px; transition: border-color 0.15s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {ACCENT}; }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: {SURFACE_3} !important; color: {TEXT} !important; border: 1px solid {BORDER} !important;
            border-radius: 9px !important; font-weight: 600 !important; box-shadow: none !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            border-color: {ACCENT} !important; color: {ACCENT} !important; background: {SURFACE_4} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button span, [data-testid="stFileUploaderDropzone"] button svg {{
            color: {TEXT} !important; fill: {TEXT} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover span, [data-testid="stFileUploaderDropzone"] button:hover svg {{
            color: {ACCENT} !important; fill: {ACCENT} !important;
        }}
        [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] div {{
            color: {TEXT_MUTED} !important;
        }}

        [data-testid="stExpander"] {{ background: {SURFACE_2}; border: 1px solid {BORDER_SOFT}; border-radius: 12px; }}
        [data-testid="stExpander"] summary {{ font-weight: 600; }}

        /* ---- metric tiles with long values ---- */
        .kp-metric-value-sm {{ font-size: 1.12rem !important; line-height: 1.3 !important; word-break: break-word;
                               overflow-wrap: anywhere; }}
        .kp-metric-delta {{ overflow-wrap: anywhere; }}

        /* ---- About page ---- */
        .kp-about-logo {{ text-align: center; padding: 0.4rem 0 0.8rem 0; }}
        .kp-about-logo img {{ margin: 0 auto !important; }}
        .kp-about-card {{ max-width: 860px; margin: 0 auto 1rem auto; }}
        .kp-about-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700;
                           color: {TEXT}; margin-bottom: 0.6rem; letter-spacing: -0.005em; }}
        .kp-about-card p {{ color: {TEXT}; line-height: 1.65; margin: 0 0 0.7rem 0; }}
        .kp-dev-card {{ max-width: 860px; margin: 0 auto; padding: 1.8rem 2rem 1.2rem 2rem; border-radius: 20px;
                        border: 1px solid {ACCENT_DIM}; position: relative; overflow: hidden;
                        background: radial-gradient(110% 150% at 0% 0%, rgba(61,214,245,0.13), rgba(61,214,245,0.02) 55%), {SURFACE_2}; }}
        .kp-dev-card::before {{ content: ""; position: absolute; left: 0; top: 0; right: 0; height: 2px;
                                background: linear-gradient(90deg, {ACCENT}, rgba(61,214,245,0)); }}
        .kp-dev-top {{ display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap; }}
        .kp-dev-mono {{ flex: 0 0 auto; position: relative; width: 108px; height: 108px; border-radius: 24px; display: flex;
                        align-items: center; justify-content: center; padding-left: 0.08em;
                        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2rem; letter-spacing: 0.08em;
                        color: #F5F7FA; background: linear-gradient(145deg, #0F2A36, #08141C);
                        border: 1px solid {ACCENT_DIM};
                        box-shadow: inset 0 0 0 6px rgba(61,214,245,0.05), 0 12px 32px rgba(61,214,245,0.16); }}
        .kp-dev-mono::before, .kp-dev-mono::after {{ content: ""; position: absolute; width: 16px; height: 16px;
                                                     border: 2px solid {ACCENT}; }}
        .kp-dev-mono::before {{ top: 9px; left: 9px; border-right: 0; border-bottom: 0; border-radius: 7px 0 0 0; }}
        .kp-dev-mono::after {{ bottom: 9px; right: 9px; border-left: 0; border-top: 0; border-radius: 0 0 7px 0; }}
        .kp-dev-id {{ flex: 1 1 320px; min-width: 0; }}
        .kp-dev-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.22em; color: {ACCENT}; }}
        .kp-dev-name {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 700; line-height: 1.12;
                        margin: 0.3rem 0 0.6rem 0; letter-spacing: -0.015em; color: #FFFFFF; }}
        .kp-dev-meta {{ display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; }}
        .kp-dev-chip {{ font-size: 0.78rem; font-weight: 600; color: #B8F1FF; padding: 0.28rem 0.8rem; border-radius: 999px;
                        border: 1px solid {ACCENT_DIM}; background: rgba(61,214,245,0.10); }}
        .kp-dev-loc {{ display: inline-flex; align-items: center; gap: 0.35rem; color: {TEXT_MUTED}; font-size: 0.88rem; }}
        .kp-dev-loc svg, .kp-dev-link svg {{ flex: 0 0 auto; }}
        .kp-dev-links {{ margin-left: auto; display: flex; gap: 0.6rem; flex-wrap: wrap; }}
        .kp-dev-link {{ display: inline-flex; align-items: center; gap: 0.45rem; text-decoration: none !important;
                        color: {TEXT} !important; font-size: 0.85rem; font-weight: 600; padding: 0.5rem 0.95rem; border-radius: 10px;
                        border: 1px solid {BORDER}; background: rgba(255,255,255,0.03); }}
        .kp-dev-link:hover {{ border-color: {ACCENT_DIM}; color: {ACCENT} !important; }}
        .kp-dev-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.4rem; margin-top: 1.5rem;
                        padding-top: 1.3rem; border-top: 1px solid {BORDER}; }}
        .kp-dev-col {{ display: flex; gap: 0.8rem; align-items: flex-start; }}
        .kp-dev-ic {{ flex: 0 0 auto; width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center;
                      justify-content: center; color: {ACCENT}; background: rgba(61,214,245,0.08); border: 1px solid {BORDER}; }}
        .kp-dev-k {{ font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 0.18em; color: {TEXT_FAINT}; }}
        .kp-dev-v {{ color: {TEXT}; font-size: 0.86rem; line-height: 1.5; margin-top: 0.25rem; }}
        .kp-dev-foot {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; justify-content: space-between; margin-top: 1.3rem;
                        padding-top: 0.9rem; border-top: 1px solid {BORDER}; color: {TEXT_FAINT};
                        font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.08em; }}
        @media (max-width: 760px) {{ .kp-dev-grid {{ grid-template-columns: 1fr; }} .kp-dev-links {{ margin-left: 0; }} }}

        .kp-footer {{ color: {TEXT_FAINT}; font-size: 0.75rem; text-align: center; padding-top: 2.4rem; letter-spacing: 0.02em; }}

        .kp-bar-track {{ background: {SURFACE_4}; border-radius: 6px; height: 7px; overflow: hidden; width: 100%; }}
        .kp-bar-fill {{ height: 100%; border-radius: 6px; }}

        .kp-divider {{ height: 1px; background: {BORDER}; margin: 1.1rem 0; border: none; }}

        @media (max-width: 640px) {{
            .block-container {{ padding-left: 0.85rem; padding-right: 0.85rem; }}
            h1 {{ font-size: 1.55rem; }}
            .kp-metric-value {{ font-size: 1.55rem; }}
            .kp-card {{ padding: 0.95rem 1.05rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )