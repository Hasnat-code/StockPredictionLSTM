# ============================================================
#  frontend.py  —  Beautiful UI only, all logic in backend.py
#  Run with:  streamlit run frontend.py
# ============================================================

import streamlit as st
import pandas as pd

# ── Import ALL logic from backend ──
from dashboard import (
    save_user, user_exists, get_all_users,
    load_master, fetch_live,
    calculate_rsi, get_signal, get_price_changes,
    run_prediction, run_validation,
    log_prediction, update_actual_prices, read_prediction_history,
    get_top_gainers, get_top_losers,
)
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="NeuralQuant — Crypto AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
#  GLOBAL CSS  —  NeuralQuant dark theme
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #0a0e1a !important;
    color: #e8eaf0 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Custom Top Bar ── */
.nq-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0 14px 0; margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.nq-logo { display: flex; align-items: center; gap: 10px; }
.nq-logo-icon {
    width: 36px; height: 36px; border-radius: 9px;
    background: linear-gradient(135deg,#00d4aa,#7c6aff);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; line-height: 1;
}
.nq-logo-text { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; color: #fff; }
.nq-logo-text span { color: #00d4aa; }
.nq-tagline { font-family: 'Space Mono', monospace; font-size: 11px; color: #6b7394; }
.nq-live {
    display: flex; align-items: center; gap: 6px;
    background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.25);
    border-radius: 20px; padding: 5px 14px;
    font-family: 'Space Mono', monospace; font-size: 11px; color: #00d4aa;
}
.nq-live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #00d4aa;
    animation: blink 1.4s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Section Headers ── */
.nq-section {
    font-size: 13px; font-weight: 700; color: #6b7394;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin: 28px 0 12px 0;
    display: flex; align-items: center; gap: 10px;
}
.nq-section::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: #181e2e !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; color: #6b7394 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 26px !important; font-weight: 700 !important; color: #fff !important;
}
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace !important; font-size: 12px !important; }

/* ── Cards / Panels ── */
.nq-card {
    background: #181e2e; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;
}
.nq-card-title {
    font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 4px;
}
.nq-card-sub {
    font-family: 'Space Mono', monospace; font-size: 11px; color: #6b7394; margin-bottom: 16px;
}

/* ── Signal Badge ── */
.nq-signal {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'Space Mono', monospace; font-size: 13px; font-weight: 700;
    padding: 10px 20px; border-radius: 10px; letter-spacing: 1px;
    margin: 8px 0;
}
.sig-buy  { background: rgba(0,212,170,0.12); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.sig-sell { background: rgba(255,94,94,0.12);  color: #ff5e5e; border: 1px solid rgba(255,94,94,0.3); }
.sig-hold { background: rgba(251,182,80,0.10); color: #fbb650; border: 1px solid rgba(251,182,80,0.3); }

/* ── RSI Label ── */
.nq-rsi-val {
    font-family: 'Space Mono', monospace; font-size: 28px; font-weight: 700;
    color: #fff; margin-top: 4px;
}
.nq-rsi-label { font-family: 'Space Mono', monospace; font-size: 10px; color: #6b7394; letter-spacing: 2px; }

/* ── Accuracy Block ── */
.nq-acc-row {
    display: flex; justify-content: space-between;
    font-family: 'Space Mono', monospace; font-size: 12px;
    padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.nq-acc-row:last-child { border-bottom: none; }
.nq-acc-key { color: #6b7394; }
.nq-acc-val { color: #fff; font-weight: 700; }

/* ── Prediction Result Box ── */
.nq-pred-box {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(124,106,255,0.06));
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 12px; padding: 20px 24px; margin-top: 12px;
}
.nq-pred-label { font-family: 'Space Mono', monospace; font-size: 10px; color: #6b7394; letter-spacing: 2px; text-transform: uppercase; }
.nq-pred-price { font-family: 'Space Mono', monospace; font-size: 36px; font-weight: 700; color: #00d4aa; margin: 6px 0; }
.nq-pred-delta { font-family: 'Space Mono', monospace; font-size: 13px; }

/* ── Backtest Box ── */
.nq-backtest-box {
    background: rgba(124,106,255,0.06); border: 1px solid rgba(124,106,255,0.2);
    border-radius: 12px; padding: 16px 20px; margin-top: 12px;
}
.nq-mape-val { font-family: 'Space Mono', monospace; font-size: 32px; font-weight: 700; color: #7c6aff; }
.nq-mape-label { font-family: 'Space Mono', monospace; font-size: 10px; color: #6b7394; letter-spacing: 2px; }

/* ── History Block ── */
.nq-history {
    background: #0f1424; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 16px; font-family: 'Space Mono', monospace;
    font-size: 11px; color: #6b7394; line-height: 1.9;
    max-height: 320px; overflow-y: auto;
    white-space: pre-wrap;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00b894) !important;
    color: #000 !important; font-weight: 800 !important;
    font-family: 'Syne', sans-serif !important;
    border: none !important; border-radius: 9px !important;
    padding: 10px 24px !important; font-size: 13px !important;
    transition: opacity .2s, transform .15s !important;
    letter-spacing: 0.4px !important;
}
.stButton > button:hover { opacity: 0.86 !important; transform: translateY(-1px) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1424 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; color: #6b7394 !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, div[data-baseweb="select"] {
    background: #141929 !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
    font-family: 'Space Mono', monospace !important; font-size: 13px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6b7394 !important;
    font-family: 'Syne', sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; border: none !important;
    padding: 10px 22px !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 20px 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* ── Progress Bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00d4aa, #7c6aff) !important;
    border-radius: 4px !important;
}
.stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius: 4px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-family: 'Space Mono', monospace !important; color: #6b7394 !important; }

/* ── Success / Error / Info / Warning ── */
.stSuccess  { background: rgba(0,212,170,0.08)  !important; border-left: 3px solid #00d4aa !important; border-radius: 8px !important; }
.stError    { background: rgba(255,94,94,0.08)   !important; border-left: 3px solid #ff5e5e !important; border-radius: 8px !important; }
.stInfo     { background: rgba(124,106,255,0.08) !important; border-left: 3px solid #7c6aff !important; border-radius: 8px !important; }
.stWarning  { background: rgba(251,182,80,0.08)  !important; border-left: 3px solid #fbb650 !important; border-radius: 8px !important; }

/* ── Movers table ── */
.nq-mover-card {
    background: #181e2e; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px 18px;
}
.nq-mover-title {
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; font-family: 'Space Mono', monospace;
    color: #6b7394; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════
for key, default in [
    ('total_preds',   0),
    ('correct_preds', 0),
    ('logged_in',     False),
    ('username',      ""),
    ('recent_stack',  []),
    ('pred_result',   None),
    ('val_result',    None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════
#  HELPER: render top bar
# ══════════════════════════════════════════════
def render_topbar(subtitle=""):
    st.markdown(f"""
    <div class="nq-topbar">
        <div class="nq-logo">
            <div class="nq-logo-icon">⚡</div>
            <div>
                <div class="nq-logo-text">Neural<span>Quant</span></div>
                <div class="nq-tagline">LSTM Crypto Intelligence {subtitle}</div>
            </div>
        </div>
        <div class="nq-live"><div class="nq-live-dot"></div> MARKET LIVE</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  LOGIN / REGISTER PAGE
# ══════════════════════════════════════════════
if not st.session_state.logged_in:

    render_topbar()

    st.markdown('<div class="nq-section">Authentication</div>', unsafe_allow_html=True)

    col_form, col_space = st.columns([1, 1.2])

    with col_form:
        st.markdown('<div class="nq-card">', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-sub">Sign in to access your LSTM predictions and history</div>', unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(["🔑 Login", "✨ Register"])

        with login_tab:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login →", key="btn_login"):
                if user_exists(login_user, login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username  = login_user
                    update_actual_prices(login_user)   # ← your original logic
                    st.success(f"Welcome back, {login_user}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with register_tab:
            reg_user = st.text_input("Create Username", key="reg_user")
            reg_pass = st.text_input("Create Password", type="password", key="reg_pass")
            if st.button("Create Account →", key="btn_register"):
                if not reg_user.strip() or not reg_pass.strip():
                    st.warning("Please fill in all fields.")
                elif reg_user in get_all_users():
                    st.error("Username already taken.")
                else:
                    save_user(reg_user, reg_pass)
                    st.success("Account created! You can now log in.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_space:
        st.markdown("""
        <div style="padding:40px 20px">
            <div style="font-size:13px;color:#6b7394;font-family:'Space Mono',monospace;line-height:2">
                <div style="color:#00d4aa;font-weight:700;font-size:15px;margin-bottom:12px">What you get</div>
                ⚡ LSTM-powered price predictions<br>
                📡 RSI Buy / Sell / Hold signals<br>
                📊 15-day backtesting validation<br>
                📜 Per-user prediction history<br>
                🔄 Auto-updated actual prices<br>
                🪙 Live crypto market data
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════
#  LOAD DATA  (cached in backend)
# ══════════════════════════════════════════════
@st.cache_data
def cached_master():
    return load_master()

@st.cache_data
def cached_live(symbol):
    return fetch_live(symbol)

master_df = cached_master()


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0">
        <div style="font-family:'Space Mono',monospace;font-size:10px;color:#6b7394;letter-spacing:2px;text-transform:uppercase">Logged in as</div>
        <div style="font-size:16px;font-weight:700;color:#00d4aa;margin-top:4px">{st.session_state.username}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("⇠ Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

    st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#6b7394;letter-spacing:2px;text-transform:uppercase;margin:20px 0 8px">Coin Directory</div>', unsafe_allow_html=True)

    search = st.text_input("Search Coin Name", "")
    filtered = master_df[master_df['Coin Name'].str.contains(search, case=False, na=False)]
    selected_name = st.selectbox("Select Asset", filtered['Coin Name'].tolist())

    selected_info = master_df[master_df['Coin Name'] == selected_name].iloc[0]
    symbol = str(selected_info['Symbol']).strip()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:11px;color:#6b7394;line-height:2">
        <div style="color:#fff;font-weight:700;margin-bottom:6px">{selected_name}</div>
        Symbol: <span style="color:#00d4aa">{symbol}</span><br>
        Stack size: <span style="color:#7c6aff">{len(st.session_state.recent_stack)}</span><br>
        Total preds: <span style="color:#fbb650">{st.session_state.total_preds}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════
render_topbar(f"· {selected_name} ({symbol})")

# ── Fetch live data ──
hist_df = cached_live(symbol)


# ── TOP MOVERS ──────────────────────────────
st.markdown('<div class="nq-section">Top Market Movers · 24h</div>', unsafe_allow_html=True)

col_g, col_l = st.columns(2)

with col_g:
    st.markdown('<div class="nq-mover-card">', unsafe_allow_html=True)
    st.markdown('<div class="nq-mover-title">🟢 Top Gainers</div>', unsafe_allow_html=True)
    st.dataframe(get_top_gainers(master_df), hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_l:
    st.markdown('<div class="nq-mover-card">', unsafe_allow_html=True)
    st.markdown('<div class="nq-mover-title">🔴 Top Losers</div>', unsafe_allow_html=True)
    st.dataframe(get_top_losers(master_df), hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


if not hist_df.empty:

    # ── PRICE INDICATORS ────────────────────────
    st.markdown('<div class="nq-section">Price Change Indicators</div>', unsafe_allow_html=True)

    prices = get_price_changes(hist_df)   # ← backend call

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price",  f"${prices['current']:,.2f}")
    m2.metric("24h Change",     f"{prices['change_24h']:.2f}%", delta=f"{prices['change_24h']:.2f}%")
    m3.metric("7d Change",      f"{prices['change_7d']:.2f}%",  delta=f"{prices['change_7d']:.2f}%")

    # ── PRICE CHART ─────────────────────────────
    st.markdown('<div class="nq-section">Historical Price Chart</div>', unsafe_allow_html=True)
    st.markdown('<div class="nq-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="nq-card-title">{selected_name} · Close Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="nq-card-sub">Daily closing prices · Source: Yahoo Finance</div>', unsafe_allow_html=True)
    st.line_chart(hist_df['Close'], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── RSI + ACCURACY ──────────────────────────
    st.markdown('<div class="nq-section">Technical Signal · Accuracy Tracker</div>', unsafe_allow_html=True)

    rsi_vals   = calculate_rsi(hist_df['Close'])    # ← backend
    latest_rsi = float(rsi_vals.iloc[-1].squeeze())
    signal     = get_signal(latest_rsi)             # ← backend

    col_sig, col_acc = st.columns(2)

    with col_sig:
        st.markdown('<div class="nq-card">', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-title">RSI Signal</div>', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-sub">Relative Strength Index · 14-period</div>', unsafe_allow_html=True)

        sig_class = {"BUY": "sig-buy", "SELL": "sig-sell", "HOLD": "sig-hold"}[signal]
        sig_icon  = {"BUY": "🟢 BUY  — Oversold Market", "SELL": "🔴 SELL — Overbought Market", "HOLD": "🟡 HOLD — Neutral Trend"}[signal]

        st.markdown(f'<div class="nq-signal {sig_class}">{sig_icon}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="nq-rsi-label">CURRENT RSI</div><div class="nq-rsi-val">{latest_rsi:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_acc:
        st.markdown('<div class="nq-card">', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-title">Accuracy Tracker</div>', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-sub">Session prediction performance</div>', unsafe_allow_html=True)

        total   = st.session_state.total_preds
        correct = st.session_state.correct_preds
        acc_pct = (correct / total * 100) if total > 0 else 0

        st.markdown(f"""
        <div class="nq-acc-row"><span class="nq-acc-key">Total Predictions</span><span class="nq-acc-val">{total}</span></div>
        <div class="nq-acc-row"><span class="nq-acc-key">Correct (MAPE &lt; 5%)</span><span class="nq-acc-val">{correct}</span></div>
        <div class="nq-acc-row"><span class="nq-acc-key">Session Accuracy</span><span class="nq-acc-val" style="color:#00d4aa">{acc_pct:.1f}%</span></div>
        """, unsafe_allow_html=True)

        st.progress(acc_pct / 100)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── PREDICTION + VALIDATION ─────────────────
    st.markdown('<div class="nq-section">LSTM Engine</div>', unsafe_allow_html=True)

    col_pred, col_val = st.columns(2)

    # ── PREDICT ──
    with col_pred:
        st.markdown('<div class="nq-card">', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-title">🤖 Tomorrow\'s Price Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-sub">Stacked LSTM · 60-day lookback · trained on full history</div>', unsafe_allow_html=True)

        if st.button("⚡ Predict Tomorrow", key="btn_predict"):
            with st.spinner("Training LSTM model… this takes a moment"):
                st.session_state.total_preds += 1
                pred = run_prediction(hist_df)          # ← backend (unchanged)
                st.session_state.recent_stack.append(pred)
                log_prediction(st.session_state.username, symbol, pred)
                st.session_state.pred_result = pred

        if st.session_state.pred_result is not None:
            pred  = st.session_state.pred_result
            diff  = pred - prices['current']
            color = "#00d4aa" if diff >= 0 else "#ff5e5e"
            arrow = "▲" if diff >= 0 else "▼"

            st.markdown(f"""
            <div class="nq-pred-box">
                <div class="nq-pred-label">AI Predicted Close Price</div>
                <div class="nq-pred-price">${pred:,.2f}</div>
                <div class="nq-pred-delta" style="color:{color}">{arrow} {diff:+,.2f} vs current</div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.recent_stack:
                latest = st.session_state.recent_stack[-1]
                st.markdown(f"""
                <div style="margin-top:10px;font-family:'Space Mono',monospace;font-size:11px;
                    color:#6b7394;padding:10px 14px;background:#0f1424;border-radius:8px;
                    border:1px solid rgba(255,255,255,0.06)">
                    Stack top → <span style="color:#7c6aff">${latest:,.2f}</span>
                    &nbsp;·&nbsp; depth: {len(st.session_state.recent_stack)}
                </div>
                """, unsafe_allow_html=True)

            st.success("Prediction logged to your history.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── VALIDATE ──
    with col_val:
        st.markdown('<div class="nq-card">', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-title">📊 Backtest Validation</div>', unsafe_allow_html=True)
        st.markdown('<div class="nq-card-sub">15-day out-of-sample test · MAPE error metric</div>', unsafe_allow_html=True)

        if st.button("📊 Validate Accuracy", key="btn_validate"):
            with st.spinner("Running backtesting engine…"):
                result = run_validation(hist_df)        # ← backend (unchanged)
                if result["is_accurate"]:
                    st.session_state.correct_preds += 1
                st.session_state.val_result = result

        if st.session_state.val_result is not None:
            result = st.session_state.val_result
            mape   = result["mape"]
            color  = "#00d4aa" if mape < 5 else "#fbb650" if mape < 10 else "#ff5e5e"

            st.markdown(f"""
            <div class="nq-backtest-box">
                <div class="nq-mape-label">AVERAGE DEVIATION (MAPE)</div>
                <div class="nq-mape-val" style="color:{color}">{mape:.2f}%</div>
                <div style="font-family:'Space Mono',monospace;font-size:11px;color:#6b7394;margin-top:6px">
                    {"✅ High accuracy — below 5% threshold" if mape < 5 else "⚠ Moderate deviation" if mape < 10 else "⛔ High deviation — retrain recommended"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:14px;font-family:\'Space Mono\',monospace;font-size:11px;color:#6b7394;letter-spacing:1px">Actual vs Predicted · Last 15 Days</div>', unsafe_allow_html=True)
            st.line_chart(result["comp_df"], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("⚠ No valid market data found for this coin. Try another.")


# ── PREDICTION HISTORY ──────────────────────
st.markdown('<div class="nq-section">Your Prediction History</div>', unsafe_allow_html=True)

history_text = read_prediction_history(st.session_state.username)   # ← backend

if history_text:
    st.markdown(f'<div class="nq-history">{history_text}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="nq-history" style="text-align:center;padding:30px">No predictions yet. Run your first LSTM prediction above!</div>', unsafe_allow_html=True)