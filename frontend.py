# ============================================================
#  frontend.py  —  NeuralQuant  (Huly-inspired)
#  Run:  streamlit run frontend.py
# ============================================================

import streamlit as st
import pandas as pd

from dashboard import (
    save_user, user_exists, get_all_users,
    load_master, fetch_live,
    calculate_rsi, get_signal, get_price_changes,
    run_prediction, run_validation,
    log_prediction, update_actual_prices, read_prediction_history,
    get_top_gainers, get_top_losers,
)

# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralQuant — LSTM Crypto AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
for k, v in [
    ('total_preds', 0), ('correct_preds', 0),
    ('logged_in', False), ('username', ""),
    ('recent_stack', []), ('pred_result', None),
    ('val_result', None), ('dark_mode', True),
]:
    if k not in st.session_state:
        st.session_state[k] = v

dark = st.session_state.dark_mode

# ──────────────────────────────────────────────
#  THEME
# ──────────────────────────────────────────────
if dark:
    BG = "#0a0e1a"; BG2 = "#0f1424"; PANEL = "#181e2e"
    BORDER = "rgba(255,255,255,0.07)"; TEXT = "#e8eaf0"; MUTED = "#6b7394"
    ACCENT = "#a78bfa"; ACCENT2 = "#7c6aff"
    UP = "#00d4aa"; DOWN = "#ff5e5e"; LIGHTNING = "#c084fc"
    TXTW = "#ffffff"
else:
    BG = "#f4f4f8"; BG2 = "#e8e8f0"; PANEL = "#ffffff"
    BORDER = "rgba(0,0,0,0.08)"; TEXT = "#1a1a2e"; MUTED = "#6b6b8a"
    ACCENT = "#7c3aed"; ACCENT2 = "#6d28d9"
    UP = "#059669"; DOWN = "#dc2626"; LIGHTNING = "#7c3aed"
    TXTW = "#1a1a2e"

# ══════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 0 2rem 0 !important; max-width:100% !important; }}
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:{BG2}; }}
::-webkit-scrollbar-thumb {{ background:rgba(167,139,250,0.3); border-radius:3px; }}

/* ── keyframes ── */
@keyframes beamPulse {{
    0%,100% {{ opacity:.8; filter:blur(1px); }}
    50%      {{ opacity:1;  filter:blur(.5px) brightness(1.4); }}
}}
@keyframes haloPulse {{
    0%,100% {{ transform:translate(-50%,-60%) scale(1);   opacity:.7; }}
    50%     {{ transform:translate(-50%,-60%) scale(1.08); opacity:1;  }}
}}
@keyframes boltFlash {{
    0%,85%,100% {{ opacity:0; }}
    88%          {{ opacity:.9; }}
    92%          {{ opacity:.2; }}
    95%          {{ opacity:.7; }}
    98%          {{ opacity:0;  }}
}}
@keyframes arcSpin {{
    from {{ transform:rotate(0deg); }}
    to   {{ transform:rotate(360deg); }}
}}
@keyframes secSpin  {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
@keyframes minSpin  {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
@keyframes hourSpin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}

/* ── HERO ── */
.nq-hero {{
    position:relative; min-height:88vh;
    background:{BG}; overflow:hidden;
    display:flex; flex-direction:column;
    justify-content:center; padding:0 6vw;
}}
.nq-beam {{
    position:absolute; top:-60px; left:50%;
    transform:translateX(-50%);
    width:2px; height:65vh;
    background:linear-gradient(180deg,transparent 0%,{ACCENT} 30%,#60a5fa 60%,rgba(167,139,250,.3) 85%,transparent 100%);
    filter:blur(1px); z-index:1;
    animation:beamPulse 3s ease-in-out infinite;
}}
.nq-halo {{
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-60%);
    width:600px; height:600px;
    background:radial-gradient(ellipse,rgba(124,106,255,.18) 0%,rgba(96,165,250,.1) 30%,transparent 70%);
    z-index:0; animation:haloPulse 4s ease-in-out infinite;
}}
.nq-hero-content {{
    position:relative; z-index:2; max-width:680px;
}}
.nq-eyebrow {{
    font-family:'Space Mono',monospace; font-size:12px;
    color:{ACCENT}; letter-spacing:3px; text-transform:uppercase;
    margin-bottom:20px; display:flex; align-items:center; gap:10px;
}}
.nq-eyebrow::before {{ content:''; width:24px; height:1px; background:{ACCENT}; }}
.nq-hero-h1 {{
    font-size:clamp(42px,5.5vw,72px); font-weight:800;
    line-height:1.04; letter-spacing:-2px;
    color:{TXTW}; margin-bottom:20px;
}}
.nq-hero-h1 em {{ font-style:normal; color:{ACCENT}; }}
.nq-hero-sub {{
    font-size:17px; color:{MUTED}; line-height:1.75;
    margin-bottom:36px; max-width:480px;
}}
.nq-pill-cta {{
    display:inline-flex; align-items:center; gap:8px;
    background:{'rgba(255,255,255,.07)' if dark else 'rgba(0,0,0,.06)'};
    border:1px solid {'rgba(255,255,255,.18)' if dark else 'rgba(0,0,0,.12)'};
    color:{TXTW}; padding:13px 30px; border-radius:30px;
    font-size:14px; font-weight:700; letter-spacing:.5px; cursor:pointer;
}}

/* ── hero frame ── */
.nq-hero-frame {{
    position:absolute; right:4vw; bottom:0;
    width:50%; max-width:680px; z-index:2;
    background:rgba(15,20,36,.88);
    border:1px solid rgba(167,139,250,.22); border-bottom:none;
    border-radius:16px 16px 0 0; padding:22px;
    backdrop-filter:blur(12px);
    box-shadow:0 0 80px rgba(124,106,255,.12);
}}
.nq-frame-dots {{ display:flex; align-items:center; gap:6px; margin-bottom:14px; }}
.nq-dot {{ width:10px; height:10px; border-radius:50%; }}
.nq-chart-ph {{
    height:200px;
    background:linear-gradient(180deg,rgba(124,106,255,.05) 0%,transparent 100%);
    border-radius:8px; border:1px dashed rgba(167,139,250,.15);
    display:flex; align-items:center; justify-content:center;
    flex-direction:column; gap:8px;
    color:{MUTED}; font-family:'Space Mono',monospace; font-size:12px;
}}

/* ── topbar ── */
.nq-topbar {{
    position:sticky; top:0; z-index:200;
    display:flex; align-items:center; justify-content:space-between;
    padding:0 32px; height:58px;
    background:{'rgba(10,14,26,.96)' if dark else 'rgba(244,244,248,.96)'};
    border-bottom:1px solid {BORDER}; backdrop-filter:blur(12px);
}}
.nq-logo-row {{ display:flex; align-items:center; gap:10px; }}
.nq-logo-icon {{
    width:34px; height:34px; border-radius:9px;
    background:linear-gradient(135deg,{ACCENT},#60a5fa);
    display:flex; align-items:center; justify-content:center; font-size:18px;
}}
.nq-logo-txt {{
    font-size:19px; font-weight:800; letter-spacing:-.5px; color:{TXTW};
}}
.nq-logo-txt em {{ font-style:normal; color:{ACCENT}; }}
.nq-nav {{ display:flex; gap:28px; }}
.nq-nav a {{
    font-size:14px; font-weight:500; color:{MUTED};
    text-decoration:none; transition:color .2s;
}}
.nq-nav a:hover, .nq-nav a.active {{ color:{TXTW}; }}
.nq-live {{
    display:flex; align-items:center; gap:6px;
    background:rgba(167,139,250,.08); border:1px solid rgba(167,139,250,.25);
    border-radius:20px; padding:5px 14px;
    font-family:'Space Mono',monospace; font-size:12px; color:{ACCENT};
}}
.live-dot {{
    width:7px; height:7px; border-radius:50%;
    background:{ACCENT}; animation:beamPulse 1.4s infinite;
}}

/* ── section label ── */
.nq-sec {{
    font-size:12px; font-weight:700; color:{MUTED};
    letter-spacing:2.5px; text-transform:uppercase;
    font-family:'Space Mono',monospace;
    margin:32px 0 16px; padding:0 32px;
    display:flex; align-items:center; gap:12px;
}}
.nq-sec::after {{ content:''; flex:1; height:1px; background:{BORDER}; }}

/* ── lightning card ── */
.lcard {{
    background:{PANEL}; border:1px solid {BORDER};
    border-radius:14px; padding:22px 24px;
    position:relative; overflow:hidden;
}}
.lcard.featured {{
    border-color:{LIGHTNING};
    box-shadow:0 0 0 1px {LIGHTNING},0 0 30px rgba(192,132,252,.22),0 0 60px rgba(192,132,252,.08);
    background:linear-gradient(135deg,rgba(124,106,255,.07) 0%,{PANEL} 60%);
}}
.lcard-bolts {{
    position:absolute; top:0; left:0; right:0; bottom:0;
    pointer-events:none; overflow:hidden; border-radius:14px;
}}
.lcard-bolts svg {{
    position:absolute; width:100%; height:100%;
    opacity:0; animation:boltFlash 4s ease-in-out infinite;
}}
.lcard-bolts svg:nth-child(2) {{ animation-delay:2s; }}
.lcard-title {{
    font-size:16px; font-weight:700; color:{TXTW};
    margin-bottom:16px; letter-spacing:.3px;
}}
.lcard-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:11px 0; border-bottom:1px solid {BORDER};
    font-family:'Space Mono',monospace;
}}
.lcard-row:last-child {{ border-bottom:none; }}
.lcard-sym {{ font-size:15px; font-weight:700; color:{TXTW}; min-width:70px; }}
.lcard-name {{ font-size:13px; color:{MUTED}; flex:1; text-align:center; }}
.lcard-chg {{ font-size:15px; font-weight:700; min-width:80px; text-align:right; }}

/* ── metrics ── */
[data-testid="metric-container"] {{
    background:{PANEL} !important; border:1px solid {BORDER} !important;
    border-radius:12px !important; padding:18px 22px !important;
    position:relative !important; overflow:hidden !important;
}}
[data-testid="metric-container"]::before {{
    content:''; position:absolute; inset:0; border-radius:12px;
    background:linear-gradient(135deg,rgba(167,139,250,.06) 0%,transparent 60%);
    pointer-events:none;
}}
[data-testid="metric-container"] label {{
    font-family:'Space Mono',monospace !important;
    font-size:11px !important; letter-spacing:1.5px !important;
    text-transform:uppercase !important; color:{MUTED} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family:'Space Mono',monospace !important;
    font-size:28px !important; font-weight:700 !important; color:{TXTW} !important;
}}
[data-testid="stMetricDelta"] {{
    font-family:'Space Mono',monospace !important; font-size:13px !important;
}}

/* ── price header banner ── */
.price-banner {{
    background:{PANEL}; border:1px solid {BORDER};
    border-radius:14px; padding:20px 24px; margin-bottom:18px;
    position:relative; overflow:hidden;
}}
.price-banner-bolts {{
    position:absolute; top:0; left:0; width:100%; height:100%;
    pointer-events:none; overflow:hidden;
}}
.price-banner-bolts svg {{
    position:absolute; width:100%; height:100%;
    opacity:0; animation:boltFlash 5s ease-in-out infinite;
}}
.price-banner-glow {{
    position:absolute; inset:0; border-radius:14px; pointer-events:none;
    background:linear-gradient(135deg,rgba(192,132,252,.05) 0%,transparent 50%,rgba(96,165,250,.05) 100%);
    animation:haloPulse 4s ease-in-out infinite;
}}
.price-banner-label {{
    font-family:'Space Mono',monospace; font-size:12px;
    color:{MUTED}; letter-spacing:2px; text-transform:uppercase;
    position:relative; z-index:1;
}}

/* ── nq-card ── */
.nq-card {{
    background:{PANEL}; border:1px solid {BORDER};
    border-radius:14px; padding:22px 24px; margin-bottom:18px;
    position:relative; overflow:hidden;
}}
.nq-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,{ACCENT},transparent); opacity:.4;
}}
.nq-card-title {{
    font-size:16px; font-weight:700; color:{TXTW}; margin-bottom:5px;
}}
.nq-card-sub {{
    font-family:'Space Mono',monospace; font-size:12px;
    color:{MUTED}; margin-bottom:16px;
}}

/* ── signal ── */
.nq-signal {{
    display:inline-flex; align-items:center; gap:8px;
    font-family:'Space Mono',monospace; font-size:15px; font-weight:700;
    padding:12px 22px; border-radius:10px; letter-spacing:1px; margin:10px 0;
}}
.sig-buy  {{ background:rgba(0,212,170,.1);  color:{UP};    border:1px solid rgba(0,212,170,.28); }}
.sig-sell {{ background:rgba(255,94,94,.1);  color:{DOWN};  border:1px solid rgba(255,94,94,.28); }}
.sig-hold {{ background:rgba(251,182,80,.1); color:#fbb650; border:1px solid rgba(251,182,80,.28); }}
.nq-rsi-label {{
    font-family:'Space Mono',monospace; font-size:11px;
    color:{MUTED}; letter-spacing:2px; margin-top:10px;
}}
.nq-rsi-val {{
    font-family:'Space Mono',monospace; font-size:34px;
    font-weight:700; color:{TXTW}; margin-top:4px;
}}

/* ── accuracy rows ── */
.acc-row {{
    display:flex; justify-content:space-between;
    font-family:'Space Mono',monospace; font-size:13px;
    padding:9px 0; border-bottom:1px solid {BORDER};
}}
.acc-row:last-child {{ border-bottom:none; }}
.acc-key {{ color:{MUTED}; }}
.acc-val {{ color:{TXTW}; font-weight:700; }}

/* ── pred box ── */
.pred-box {{
    background:linear-gradient(135deg,rgba(167,139,250,.09),rgba(96,165,250,.05));
    border:1px solid rgba(167,139,250,.3);
    border-radius:12px; padding:22px 26px; margin-top:14px;
}}
.pred-label {{
    font-family:'Space Mono',monospace; font-size:11px;
    color:{MUTED}; letter-spacing:2px; text-transform:uppercase;
}}
.pred-price {{
    font-family:'Space Mono',monospace; font-size:38px;
    font-weight:700; color:{ACCENT}; margin:8px 0;
}}
.pred-delta {{ font-family:'Space Mono',monospace; font-size:14px; }}

/* ── backtest box ── */
.bt-box {{
    background:rgba(124,106,255,.07); border:1px solid rgba(124,106,255,.22);
    border-radius:12px; padding:18px 22px; margin-top:14px;
}}
.bt-mape {{ font-family:'Space Mono',monospace; font-size:34px; font-weight:700; }}
.bt-label {{ font-family:'Space Mono',monospace; font-size:11px; color:{MUTED}; letter-spacing:2px; }}

/* ── history ── */
.nq-history {{
    background:{BG2}; border:1px solid {BORDER}; border-radius:10px;
    padding:18px; font-family:'Space Mono',monospace; font-size:12px;
    color:{MUTED}; line-height:2; max-height:340px; overflow-y:auto; white-space:pre-wrap;
}}

/* ── buttons ── */
.stButton > button {{
    background:linear-gradient(135deg,{ACCENT},{ACCENT2}) !important;
    color:#fff !important; font-weight:800 !important;
    font-family:'Syne',sans-serif !important; border:none !important;
    border-radius:9px !important; padding:11px 26px !important;
    font-size:14px !important; letter-spacing:.4px !important;
    transition:opacity .2s,transform .15s !important;
}}
.stButton > button:hover {{ opacity:.86 !important; transform:translateY(-1px) !important; }}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background:{BG2} !important; border-right:1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] label {{
    font-family:'Space Mono',monospace !important;
    font-size:11px !important; letter-spacing:1.5px !important;
    text-transform:uppercase !important; color:{MUTED} !important;
}}

/* ── inputs ── */
.stTextInput input, div[data-baseweb="select"] {{
    background:{BG2} !important; border:1px solid {BORDER} !important;
    border-radius:8px !important; color:{TEXT} !important;
    font-family:'Space Mono',monospace !important; font-size:13px !important;
}}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:transparent !important;
    border-bottom:1px solid {BORDER} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{MUTED} !important;
    font-family:'Syne',sans-serif !important; font-size:14px !important;
    font-weight:600 !important; border:none !important; padding:10px 22px !important;
}}
.stTabs [aria-selected="true"] {{
    color:{ACCENT} !important; border-bottom:2px solid {ACCENT} !important;
}}

/* ── progress ── */
.stProgress > div > div {{
    background:linear-gradient(90deg,{ACCENT},#60a5fa) !important; border-radius:4px !important;
}}
.stProgress > div {{ background:rgba(255,255,255,.06) !important; border-radius:4px !important; }}

/* ── alerts ── */
.stSuccess  {{ background:rgba(0,212,170,.07)  !important; border-left:3px solid {UP}    !important; border-radius:8px !important; }}
.stError    {{ background:rgba(255,94,94,.07)   !important; border-left:3px solid {DOWN}  !important; border-radius:8px !important; }}
.stInfo     {{ background:rgba(167,139,250,.07) !important; border-left:3px solid {ACCENT}!important; border-radius:8px !important; }}
.stWarning  {{ background:rgba(251,182,80,.07)  !important; border-left:3px solid #fbb650 !important; border-radius:8px !important; }}
hr {{ border-color:{BORDER} !important; margin:20px 0 !important; }}

/* ── clock section ── */
.clock-section {{
    position:relative; min-height:340px;
    background:{BG2}; border-top:1px solid {BORDER};
    display:flex; align-items:center; justify-content:center;
    gap:80px; overflow:hidden; padding:60px 32px;
}}
.clock-section::before {{
    content:''; position:absolute; top:-40%; left:-10%;
    width:55%; height:180%;
    background:linear-gradient(125deg,rgba(234,88,12,.18) 0%,rgba(124,106,255,.12) 40%,transparent 70%);
    transform:skewX(-15deg); pointer-events:none;
}}
.clock-face {{
    width:220px; height:220px; border-radius:50%; flex-shrink:0;
    background:radial-gradient(circle,#1c1c2e 60%,#0f0f1a 100%);
    border:2px solid rgba(255,255,255,.08);
    box-shadow:0 0 0 8px rgba(0,0,0,.4),0 0 40px rgba(124,106,255,.2);
    position:relative; overflow:hidden;
}}
.clock-arc-o {{
    position:absolute; inset:-3px; border-radius:50%;
    background:conic-gradient(from 0deg,transparent 0%,rgba(234,88,12,.8) 25%,transparent 30%);
    animation:arcSpin 8s linear infinite;
}}
.clock-arc-b {{
    position:absolute; inset:-3px; border-radius:50%;
    background:conic-gradient(from 180deg,transparent 0%,rgba(96,165,250,.8) 20%,{ACCENT} 25%,transparent 30%);
    animation:arcSpin 8s linear infinite;
}}
.clock-inner {{
    position:absolute; inset:6px; border-radius:50%;
    background:#12121e;
    display:flex; align-items:center; justify-content:center;
}}
.clock-center {{ position:relative; width:100%; height:100%; }}
.hand {{
    position:absolute; bottom:50%; left:50%;
    transform-origin:bottom center; border-radius:4px;
}}
.hand-h {{ width:4px; height:55px; margin-left:-2px; background:rgba(255,255,255,.9); }}
.hand-m {{ width:3px; height:72px; margin-left:-1.5px; background:rgba(255,255,255,.7); }}
.hand-s {{ width:2px; height:78px; margin-left:-1px; background:{ACCENT}; }}
.hand-dot {{
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    width:10px; height:10px; border-radius:50%;
    background:{ACCENT}; box-shadow:0 0 10px {ACCENT}; z-index:10;
}}
.cta-wrap {{ max-width:420px; }}
.cta-eye {{
    font-family:'Space Mono',monospace; font-size:11px;
    color:{ACCENT}; letter-spacing:3px; text-transform:uppercase; margin-bottom:14px;
}}
.cta-h2 {{
    font-size:clamp(32px,4vw,52px); font-weight:800;
    line-height:1.05; letter-spacing:-1.5px; color:{TXTW}; margin-bottom:14px;
}}
.cta-h2 em {{ font-style:normal; color:{ACCENT}; }}
.cta-sub {{ font-size:15px; color:{MUTED}; line-height:1.7; margin-bottom:24px; }}
.cta-btns {{ display:flex; gap:12px; flex-wrap:wrap; }}
.btn-outline {{
    display:inline-flex; align-items:center; gap:6px;
    background:{'rgba(255,255,255,.06)' if dark else 'rgba(0,0,0,.05)'};
    border:1px solid {'rgba(255,255,255,.2)' if dark else 'rgba(0,0,0,.15)'};
    color:{TXTW}; padding:11px 26px; border-radius:30px;
    font-size:14px; font-weight:600; cursor:pointer; font-family:'Syne',sans-serif;
}}
.btn-solid {{
    display:inline-flex; align-items:center; gap:6px;
    background:{ACCENT}; border:1px solid {ACCENT};
    color:#fff; padding:11px 26px; border-radius:30px;
    font-size:14px; font-weight:700; cursor:pointer; font-family:'Syne',sans-serif;
}}

/* ── footer ── */
.nq-footer {{
    background:{BG}; border-top:1px solid {BORDER};
    padding:18px 32px; display:flex;
    justify-content:space-between; align-items:center;
}}
.nq-footer-l {{ font-family:'Space Mono',monospace; font-size:12px; color:{MUTED}; }}
.nq-footer-r {{ display:flex; gap:20px; }}
.nq-footer-r a {{ font-size:12px; color:{MUTED}; text-decoration:none; }}
.nq-footer-r a:hover {{ color:{ACCENT}; }}

/* pad wrapper */
.cp {{ padding:0 32px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DARK MODE TOGGLE (sidebar)
# ══════════════════════════════════════════════
with st.sidebar:
    lbl = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(lbl, key="toggle_mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# ══════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{ display:none !important; }}
    .block-container {{ padding:0 !important; }}
    </style>
    <div style="min-height:100vh;background:{BG};display:flex;flex-direction:column;
        align-items:center;justify-content:center;position:relative;overflow:hidden;">
        <div style="position:fixed;top:-80px;left:50%;transform:translateX(-50%);
            width:2px;height:60vh;
            background:linear-gradient(180deg,transparent 0%,{ACCENT} 40%,#60a5fa 70%,transparent 100%);
            filter:blur(1px);animation:beamPulse 3s ease-in-out infinite;z-index:0;"></div>
        <div style="position:fixed;top:30%;left:50%;transform:translate(-50%,-50%);
            width:500px;height:500px;
            background:radial-gradient(ellipse,rgba(124,106,255,.14) 0%,transparent 70%);
            z-index:0;animation:haloPulse 4s ease-in-out infinite;"></div>
        <div style="position:relative;z-index:2;
            background:{'rgba(24,30,46,.88)' if dark else 'rgba(255,255,255,.94)'};
            border:1px solid {'rgba(167,139,250,.22)' if dark else 'rgba(0,0,0,.1)'};
            border-radius:20px;padding:42px 46px;width:100%;max-width:420px;
            backdrop-filter:blur(16px);box-shadow:0 8px 60px rgba(0,0,0,.4);">
            <div style="display:flex;align-items:center;gap:12px;justify-content:center;margin-bottom:10px;">
                <div style="width:42px;height:42px;border-radius:11px;
                    background:linear-gradient(135deg,{ACCENT},#60a5fa);
                    display:flex;align-items:center;justify-content:center;font-size:22px;">⚡</div>
                <div style="font-size:24px;font-weight:800;letter-spacing:-.5px;color:{TXTW};">
                    Neural<span style="color:{ACCENT};">Quant</span></div>
            </div>
            <div style="text-align:center;font-family:'Space Mono',monospace;font-size:12px;
                color:{MUTED};margin-bottom:30px;letter-spacing:.5px;">
                LSTM-powered crypto intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        lt, rt = st.tabs(["🔑  Login", "✨  Register"])
        with lt:
            lu = st.text_input("Username", key="login_user", placeholder="your username")
            lp = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            if st.button("Sign In →", key="btn_login"):
                if user_exists(lu, lp):
                    st.session_state.logged_in = True
                    st.session_state.username  = lu
                    update_actual_prices(lu)
                    st.success(f"Welcome back, {lu}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with rt:
            ru = st.text_input("Create Username", key="reg_user", placeholder="choose a username")
            rp = st.text_input("Create Password", type="password", key="reg_pass", placeholder="••••••••")
            if st.button("Create Account →", key="btn_register"):
                if not ru.strip() or not rp.strip():
                    st.warning("Please fill in all fields.")
                elif ru in get_all_users():
                    st.error("Username already taken.")
                else:
                    save_user(ru, rp)
                    st.success("Account created! Sign in above.")
    st.stop()


# ══════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════
@st.cache_data
def cached_master():
    return load_master()

@st.cache_data
def cached_live(sym):
    return fetch_live(sym)

master_df = cached_master()


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px">
        <div style="font-family:'Space Mono',monospace;font-size:10px;
            color:{MUTED};letter-spacing:2px;text-transform:uppercase;">Logged in as</div>
        <div style="font-size:17px;font-weight:700;color:{ACCENT};margin-top:4px;">
            {st.session_state.username}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("⇠ Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()
    st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:{MUTED};letter-spacing:2px;text-transform:uppercase;margin:20px 0 8px;">Coin Directory</div>', unsafe_allow_html=True)
    search       = st.text_input("Search Coin", "")
    filtered     = master_df[master_df['Coin Name'].str.contains(search, case=False, na=False)]
    selected_name= st.selectbox("Select Asset", filtered['Coin Name'].tolist())
    sel_info     = master_df[master_df['Coin Name'] == selected_name].iloc[0]
    symbol       = str(sel_info['Symbol']).strip()
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};line-height:2.2;">
        <div style="color:{TXTW};font-weight:700;margin-bottom:6px;">{selected_name}</div>
        Symbol: <span style="color:{ACCENT};">{symbol}</span><br>
        Stack: <span style="color:{ACCENT2};">{len(st.session_state.recent_stack)}</span><br>
        Preds: <span style="color:#fbb650;">{st.session_state.total_preds}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TOPBAR
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="nq-topbar">
    <div class="nq-logo-row">
        <div class="nq-logo-icon">⚡</div>
        <div class="nq-logo-txt">Neural<em>Quant</em></div>
    </div>
    <div class="nq-nav">
        <a class="active">Dashboard</a>
        <a>Predictions</a>
        <a>LSTM Models</a>
        <a>Portfolio</a>
        <a>History</a>
    </div>
    <div class="nq-live"><span class="live-dot"></span> MARKET LIVE</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="nq-hero">
    <div class="nq-beam"></div>
    <div class="nq-halo"></div>
    <div class="nq-hero-content">
        <div class="nq-eyebrow">LSTM Neural Intelligence</div>
        <h1 class="nq-hero-h1">Neural<em>Quant</em><br>Predict Before<br>the Market Moves</h1>
        <p class="nq-hero-sub">
            Advanced LSTM deep learning on live crypto data —
            RSI signals, price forecasts, and backtested accuracy in real time.
        </p>
        <span class="nq-pill-cta">SEE IN ACTION &nbsp;→</span>
    </div>
    <div class="nq-hero-frame">
        <div class="nq-frame-dots">
            <div class="nq-dot" style="background:#ff5f57;"></div>
            <div class="nq-dot" style="background:#febc2e;"></div>
            <div class="nq-dot" style="background:#28c840;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:12px;
                color:{MUTED};margin-left:8px;">{selected_name} · Live Chart</span>
        </div>
        <div class="nq-chart-ph">
            <span style="font-size:24px;">📈</span>
            <span>// your live chart renders here</span>
            <span style="opacity:.5;font-size:11px;">st.line_chart · plotly · d3</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  FETCH DATA
# ══════════════════════════════════════════════
hist_df = cached_live(symbol)


# ══════════════════════════════════════════════
#  TOP MOVERS — lightning cards built with st.markdown directly in columns
# ══════════════════════════════════════════════
st.markdown('<div class="nq-sec">Top Market Movers · 24h</div>', unsafe_allow_html=True)

gainers = get_top_gainers(master_df)
losers  = get_top_losers(master_df)

st.markdown('<div class="cp">', unsafe_allow_html=True)
col_g, col_l = st.columns(2)

# ── build gainers rows HTML ──
g_rows = ""
for _, row in gainers.iterrows():
    g_rows += f"""
    <div class="lcard-row">
        <span class="lcard-sym">{str(row['Symbol'])[:8]}</span>
        <span class="lcard-name">{str(row['Coin Name'])[:20]}</span>
        <span class="lcard-chg" style="color:{UP};">+{float(row['24h']):.2f}%</span>
    </div>"""

# ── build losers rows HTML ──
l_rows = ""
for _, row in losers.iterrows():
    l_rows += f"""
    <div class="lcard-row">
        <span class="lcard-sym">{str(row['Symbol'])[:8]}</span>
        <span class="lcard-name">{str(row['Coin Name'])[:20]}</span>
        <span class="lcard-chg" style="color:{DOWN};">{float(row['24h']):.2f}%</span>
    </div>"""

with col_g:
    # NOTE: st.markdown with unsafe_allow_html=True renders HTML correctly in columns
    st.markdown(f"""
<div class="lcard featured">
    <div class="lcard-bolts">
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="2,0 2,80 8,110 2,140 2,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
            <polyline points="398,0 398,60 392,100 398,130 392,180 398,230 398,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
        </svg>
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="2,0 2,50 9,90 2,120 9,160 2,200 2,300" stroke="{LIGHTNING}" stroke-width="1.5" fill="none"/>
            <polyline points="398,0 398,80 391,120 398,150 398,300" stroke="{LIGHTNING}" stroke-width="1.5" fill="none"/>
        </svg>
    </div>
    <div class="lcard-title">🟢 Top Gainers</div>
    {g_rows}
</div>
""", unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
<div class="lcard">
    <div class="lcard-bolts">
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="2,0 2,80 8,110 2,140 2,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
            <polyline points="398,0 398,60 392,100 398,130 392,180 398,230 398,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
        </svg>
    </div>
    <div class="lcard-title">🔴 Top Losers</div>
    {l_rows}
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
if not hist_df.empty:

    prices = get_price_changes(hist_df)

    # ── PRICE INDICATORS ──
    st.markdown('<div class="nq-sec">Price Change Indicators</div>', unsafe_allow_html=True)
    st.markdown('<div class="cp">', unsafe_allow_html=True)

    # Lightning banner ABOVE the metrics
    st.markdown(f"""
<div class="price-banner">
    <div class="price-banner-glow"></div>
    <div class="price-banner-bolts">
        <svg viewBox="0 0 900 80" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="0,0 0,25 6,42 0,58 0,80" stroke="{LIGHTNING}" stroke-width="1.5" fill="none"/>
            <polyline points="900,0 900,30 894,50 900,65 900,80" stroke="{LIGHTNING}" stroke-width="1.5" fill="none"/>
        </svg>
    </div>
    <div class="price-banner-label">Live Market Data · {selected_name} ({symbol})</div>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price",  f"${prices['current']:,.2f}")
    m2.metric("24h Change",     f"{prices['change_24h']:.2f}%", delta=f"{prices['change_24h']:.2f}%")
    m3.metric("7d Change",      f"{prices['change_7d']:.2f}%",  delta=f"{prices['change_7d']:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PRICE CHART ──
    st.markdown('<div class="nq-sec">Historical Price Chart</div>', unsafe_allow_html=True)
    st.markdown('<div class="cp">', unsafe_allow_html=True)
    st.markdown(f"""
<div class="nq-card">
    <div class="nq-card-title">{selected_name} · Close Price</div>
    <div class="nq-card-sub">Daily closing prices · Source: Yahoo Finance</div>
</div>
""", unsafe_allow_html=True)
    st.line_chart(hist_df['Close'], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── RSI + ACCURACY ──
    st.markdown('<div class="nq-sec">Technical Signal · Accuracy Tracker</div>', unsafe_allow_html=True)

    rsi_vals   = calculate_rsi(hist_df['Close'])
    latest_rsi = float(rsi_vals.iloc[-1].squeeze())
    signal     = get_signal(latest_rsi)
    sig_cls    = {"BUY":"sig-buy","SELL":"sig-sell","HOLD":"sig-hold"}[signal]
    sig_txt    = {"BUY":"🟢 BUY — Oversold","SELL":"🔴 SELL — Overbought","HOLD":"🟡 HOLD — Neutral"}[signal]

    total   = st.session_state.total_preds
    correct = st.session_state.correct_preds
    acc_pct = (correct / total * 100) if total > 0 else 0

    st.markdown('<div class="cp">', unsafe_allow_html=True)
    cs, ca = st.columns(2)

    with cs:
        st.markdown(f"""
<div class="nq-card">
    <div class="nq-card-title">RSI Signal</div>
    <div class="nq-card-sub">Relative Strength Index · 14-period</div>
    <div class="nq-signal {sig_cls}">{sig_txt}</div>
    <div class="nq-rsi-label">CURRENT RSI</div>
    <div class="nq-rsi-val">{latest_rsi:.2f}</div>
</div>
""", unsafe_allow_html=True)

    with ca:
        st.markdown(f"""
<div class="nq-card">
    <div class="nq-card-title">Accuracy Tracker</div>
    <div class="nq-card-sub">Session prediction performance</div>
    <div class="acc-row"><span class="acc-key">Total Predictions</span><span class="acc-val">{total}</span></div>
    <div class="acc-row"><span class="acc-key">Correct (MAPE &lt; 5%)</span><span class="acc-val">{correct}</span></div>
    <div class="acc-row"><span class="acc-key">Session Accuracy</span>
        <span class="acc-val" style="color:{ACCENT};">{acc_pct:.1f}%</span></div>
</div>
""", unsafe_allow_html=True)
        st.progress(acc_pct / 100)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── LSTM ENGINE ──
    st.markdown('<div class="nq-sec">LSTM Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="cp">', unsafe_allow_html=True)
    cp, cv = st.columns(2)

    with cp:
        st.markdown(f"""
<div class="nq-card">
    <div class="nq-card-title">🤖 Tomorrow's Prediction</div>
    <div class="nq-card-sub">Bidirectional LSTM · 60-day lookback · 16 features</div>
</div>
""", unsafe_allow_html=True)
        if st.button("⚡ Predict Tomorrow", key="btn_predict"):
            with st.spinner("Training LSTM model…"):
                st.session_state.total_preds += 1
                pred = run_prediction(hist_df)
                st.session_state.recent_stack.append(pred)
                log_prediction(st.session_state.username, symbol, pred)
                st.session_state.pred_result = pred

        if st.session_state.pred_result is not None:
            pred  = st.session_state.pred_result
            diff  = pred - prices['current']
            clr   = UP if diff >= 0 else DOWN
            arrow = "▲" if diff >= 0 else "▼"
            st.markdown(f"""
<div class="pred-box">
    <div class="pred-label">AI Predicted Close Price</div>
    <div class="pred-price">${pred:,.2f}</div>
    <div class="pred-delta" style="color:{clr};">{arrow} {diff:+,.2f} vs current</div>
</div>
""", unsafe_allow_html=True)
            if st.session_state.recent_stack:
                latest = st.session_state.recent_stack[-1]
                st.markdown(f"""
<div style="margin-top:10px;font-family:'Space Mono',monospace;font-size:12px;
    color:{MUTED};padding:10px 14px;background:{BG2};
    border-radius:8px;border:1px solid {BORDER};">
    Stack top → <span style="color:{ACCENT2};">${latest:,.2f}</span>
    &nbsp;·&nbsp; depth: {len(st.session_state.recent_stack)}
</div>
""", unsafe_allow_html=True)
            st.success("Prediction logged to your history.")

    with cv:
        st.markdown(f"""
<div class="nq-card">
    <div class="nq-card-title">📊 Backtest Validation</div>
    <div class="nq-card-sub">20-day out-of-sample · MAPE error metric</div>
</div>
""", unsafe_allow_html=True)
        if st.button("📊 Validate Accuracy", key="btn_validate"):
            with st.spinner("Running backtesting engine…"):
                result = run_validation(hist_df)
                if result["is_accurate"]:
                    st.session_state.correct_preds += 1
                st.session_state.val_result = result

        if st.session_state.val_result is not None:
            result  = st.session_state.val_result
            mape    = result["mape"]
            mclr    = UP if mape < 5 else "#fbb650" if mape < 10 else DOWN
            verdict = "✅ High accuracy — below 5%" if mape < 5 else "⚠ Moderate deviation" if mape < 10 else "⛔ High deviation"
            st.markdown(f"""
<div class="bt-box">
    <div class="bt-label">AVERAGE DEVIATION (MAPE)</div>
    <div class="bt-mape" style="color:{mclr};">{mape:.2f}%</div>
    <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};margin-top:6px;">{verdict}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:14px;font-family:\'Space Mono\',monospace;font-size:12px;color:{MUTED};letter-spacing:1px;">Actual vs Predicted · Last 20 Days</div>', unsafe_allow_html=True)
            st.line_chart(result["comp_df"], use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="cp">', unsafe_allow_html=True)
    st.warning("⚠ No valid market data found for this coin.")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PREDICTION HISTORY
# ══════════════════════════════════════════════
st.markdown('<div class="nq-sec">Your Prediction History</div>', unsafe_allow_html=True)
st.markdown('<div class="cp">', unsafe_allow_html=True)
history_text = read_prediction_history(st.session_state.username)
if history_text:
    st.markdown(f'<div class="nq-history">{history_text}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="nq-history" style="text-align:center;padding:30px;">No predictions yet. Run your first LSTM prediction above!</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  CLOCK + CTA  (image 2 style)
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="clock-section">
    <div class="clock-face">
        <div class="clock-arc-o"></div>
        <div class="clock-arc-b"></div>
        <div class="clock-inner">
            <div class="clock-center">
                <div class="hand hand-h" id="hh"></div>
                <div class="hand hand-m" id="hm"></div>
                <div class="hand hand-s" id="hs"></div>
                <div class="hand-dot"></div>
            </div>
        </div>
    </div>
    <div class="cta-wrap">
        <div class="cta-eye">Start your journey</div>
        <div class="cta-h2">Trade Smarter.<br>Predict <em>Earlier.</em></div>
        <div class="cta-sub">
            NeuralQuant uses deep LSTM networks trained on live market data
            to give you an edge before the market moves.
            This journey is just getting started.
        </div>
        <div class="cta-btns">
            <span class="btn-outline">⚡ See in Action</span>
            <span class="btn-solid">+ Run LSTM Now</span>
        </div>
    </div>
</div>
<script>
(function() {{
    function tickClock() {{
        var now = new Date();
        var s = now.getSeconds(), m = now.getMinutes(), h = now.getHours() % 12;
        var sd = s * 6, md = m * 6 + s * 0.1, hd = h * 30 + m * 0.5;
        var eh = document.getElementById('hh');
        var em = document.getElementById('hm');
        var es = document.getElementById('hs');
        if(eh) eh.style.transform = 'rotate(' + hd + 'deg)';
        if(em) em.style.transform = 'rotate(' + md + 'deg)';
        if(es) es.style.transform = 'rotate(' + sd + 'deg)';
    }}
    tickClock();
    setInterval(tickClock, 1000);
}})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="nq-footer">
    <div class="nq-footer-l">
        © 2026 NeuralQuant &nbsp;·&nbsp; LSTM v2.4 &nbsp;·&nbsp;
        Data: Yahoo Finance &nbsp;·&nbsp; Not financial advice
    </div>
    <div class="nq-footer-r">
        <a>Documentation</a>
        <a>API Reference</a>
        <a>Model Cards</a>
        <a>Privacy</a>
        <a style="color:{ACCENT};">♥ Made with NeuralQuant</a>
    </div>
</div>
""", unsafe_allow_html=True)