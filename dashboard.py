# ============================================================
#  dashboard.py  —  Backend v4.0
#
#  ACCURACY IMPROVEMENTS vs v3:
#   1. fetch_live        → start 2017-01-01 (3× more training data)
#   2. fetch_fear_greed  → Fear & Greed Index (alternative.me, free)
#   3. fetch_btc_dominance → BTC dominance proxy (independent signal)
#   4. build_features    → 28 features, KEY FIXES:
#       • Log_Return added  → model predicts CHANGE not level (fixes lag)
#       • BB_PctB added     → where is price in the band?
#       • Price_vs_SMA50/200 → relative position signals
#       • Vol_Ratio instead of raw Vol_SMA
#       • Return_3d/7d/14d/30d → multi-timeframe momentum
#       • HL_Range → intraday range (volatility signal)
#       • Fear_Greed → sentiment (external, independent)
#       • BTC_Dom → market context (external, independent)
#       • CSV_Rank/Mom_24h/7d/30d/VolRank → from CryptocurrencyData.csv
#       • REMOVED: BB_Upper, BB_Lower, EMA_10, MACD, MACD_Signal (redundant)
#   5. make_sequences    → y = next-day Log_Return (not raw Close)
#   6. run_prediction    → converts predicted log_return → price
#      pred_price = last_close × exp(predicted_log_return)
# ============================================================

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from collections import deque
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

USERS_FILE = "users.txt"


# ──────────────────────────────────────────────
#  AUTH
# ──────────────────────────────────────────────
def save_user(username: str, password: str) -> None:
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            f.write("Username|Password\n")
    with open(USERS_FILE, "a") as f:
        f.write(f"{username}|{password}\n")


def user_exists(username: str, password: str) -> bool:
    if not os.path.exists(USERS_FILE):
        return False
    with open(USERS_FILE, "r") as f:
        for line in f.readlines():
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) == 2 and parts[0] == username and parts[1] == password:
                    return True
    return False


def get_all_users() -> set:
    users = set()
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            for line in f.readlines()[1:]:
                if "|" in line:
                    users.add(line.strip().split("|")[0])
    return users


# ──────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────
def clean_val(x):
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '').replace('%', '').strip()
        if clean_str in ['', '-']:
            return np.nan
        return float(clean_str)
    return x


# ──────────────────────────────────────────────
#  PREDICTION HISTORY
# ──────────────────────────────────────────────
def log_prediction(username: str, coin: str, pred: float, actual: str = "Pending") -> None:
    file_path    = f"{username}_prediction_history.txt"
    display_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    today_str    = date.today().isoformat()
    new_entry = (
        f"{display_date} | {coin} | Prediction: {pred:,.2f} | "
        f"Actual: {actual} | pred_date:{today_str}\n"
    )
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(f"Prediction History of {username}\n")
            f.write("-" * 60 + "\n")
    with open(file_path, "a") as f:
        f.write(new_entry)


def update_actual_prices(username: str) -> None:
    """Only replaces Pending if prediction was made on a previous calendar day."""
    file_path = f"{username}_prediction_history.txt"
    if not os.path.exists(file_path):
        return
    today_str     = date.today().isoformat()
    updated_lines = deque()
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Actual: Pending" in line:
            try:
                pred_date_str = ""
                if "pred_date:" in line:
                    pred_date_str = line.split("pred_date:")[-1].strip()
                if pred_date_str and pred_date_str >= today_str:
                    updated_lines.append(line)
                    continue
                parts  = line.split("|")
                coin   = parts[1].strip()
                ticker = f"{coin}-USD"
                latest = yf.download(ticker, period="2d", progress=False)
                if not latest.empty:
                    actual_price = float(latest['Close'].iloc[-1].squeeze())
                    clean_line   = line.split("| pred_date:")[0].strip()
                    clean_line   = clean_line.replace(
                        "Actual: Pending", f"Actual: {actual_price:,.2f}"
                    ) + "\n"
                    line = clean_line
            except Exception:
                pass
        else:
            if "pred_date:" in line:
                line = line.split("| pred_date:")[0].strip() + "\n"
        updated_lines.append(line)
    with open(file_path, "w") as f:
        f.writelines(updated_lines)


def read_prediction_history(username: str) -> str:
    file_path = f"{username}_prediction_history.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return ""


# ──────────────────────────────────────────────
#  DELETE PREDICTION HISTORY
# ──────────────────────────────────────────────
def delete_prediction_history(username: str) -> bool:
    """
    Deletes the user's prediction history file.
    Returns True if deleted successfully,
    False if file does not exist.
    """

    file_path = f"{username}_prediction_history.txt"

    if os.path.exists(file_path):
        os.remove(file_path)
        return True

    return False


# ──────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────
def load_master() -> pd.DataFrame:
    df = pd.read_csv("CryptocurrencyData.csv")
    df.columns = df.columns.str.strip()
    for col in ['Price', '1h', '24h', '7d', '30d', '24h Volume', 'Market Cap']:
        if col in df.columns:
            df[col] = df[col].apply(clean_val)
    return df


def get_csv_context(symbol: str, master_df: pd.DataFrame) -> dict:
    """
    Extract static context features for a coin from CryptocurrencyData.csv.
    Returns normalised rank, momentum %, volume rank.
    Safe — returns neutral values if coin not found.
    """
    row = master_df[master_df['Symbol'].str.strip().str.upper() == symbol.upper()]
    if row.empty:
        return {"rank_norm": 0.5, "chg_24h": 0.0, "chg_7d": 0.0,
                "chg_30d": 0.0, "vol_rank_norm": 0.5}
    r         = row.iloc[0]
    total     = max(len(master_df), 1)
    rank_norm = 1.0 - (float(r.get('Rank', total)) / total)

    vols     = master_df['24h Volume'].dropna()
    coin_vol = float(r.get('24h Volume', 0) or 0)
    vol_rank = float((vols < coin_vol).sum()) / max(len(vols), 1)

    return {
        "rank_norm":     float(np.clip(rank_norm, 0, 1)),
        "chg_24h":       float(r.get('24h',  0) or 0),
        "chg_7d":        float(r.get('7d',   0) or 0),
        "chg_30d":       float(r.get('30d',  0) or 0),
        "vol_rank_norm": float(np.clip(vol_rank, 0, 1)),
    }


# ──────────────────────────────────────────────
#  EXTERNAL SIGNAL: FEAR & GREED INDEX
#  Free — alternative.me — no API key needed.
# ──────────────────────────────────────────────
def fetch_fear_greed(n_days: int = 3000) -> pd.Series:
    """
    Returns pd.Series indexed by date, values 0–1.
    Falls back to empty Series on network failure.
    """
    try:
        url  = f"https://api.alternative.me/fng/?limit={n_days}&format=json"
        resp = requests.get(url, timeout=8)
        data = resp.json().get("data", [])
        if not data:
            return pd.Series(dtype=float)
        dates  = pd.to_datetime([d["timestamp"] for d in data], unit='s')
        values = [float(d["value"]) / 100.0 for d in data]
        return pd.Series(values, index=dates).sort_index()
    except Exception:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────
#  EXTERNAL SIGNAL: BTC DOMINANCE PROXY
#  BTC / (BTC + ETH supply-weighted) ratio.
#  Aligned to the coin's date index.
# ──────────────────────────────────────────────
def fetch_btc_dominance(index: pd.DatetimeIndex) -> pd.Series:
    """
    Returns pd.Series of BTC dominance proxy (0–1), aligned to index.
    Falls back to 0.45 on error.
    """
    try:
        btc = yf.download("BTC-USD", start="2017-01-01",
                          auto_adjust=True, progress=False)['Close']
        eth = yf.download("ETH-USD", start="2017-01-01",
                          auto_adjust=True, progress=False)['Close']
        if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
        if isinstance(eth, pd.DataFrame): eth = eth.iloc[:, 0]
        btc  = btc.squeeze().reindex(index, method='ffill')
        eth  = eth.squeeze().reindex(index, method='ffill').fillna(1.0)
        dom  = btc / (btc + eth * 120.0)   # rough supply-weighted
        return dom.clip(0, 1).fillna(0.45)
    except Exception:
        return pd.Series(0.45, index=index)


# ──────────────────────────────────────────────
#  FETCH LIVE — start 2017 for max training data
# ──────────────────────────────────────────────
def fetch_live(coin_symbol: str) -> pd.DataFrame:
    ticker = f"{coin_symbol}-USD"
    data = yf.download(
        ticker,
        start="2017-01-01",
        end=datetime.now(),
        auto_adjust=True,
        progress=False
    )
    if data.empty:
        return pd.DataFrame()
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()
    return df


# ──────────────────────────────────────────────
#  FEATURE ENGINEERING — 28 features
#
#  THE LAG FIX:
#  Old approach: predict raw Close → model learns
#  "tomorrow ≈ today" (trivially true, useless).
#
#  New approach: first column = Log_Return.
#  Sequences target the NEXT row's Log_Return.
#  Model must learn: "given 90 days of signals,
#  will tomorrow's return be positive or negative
#  and by how much?" — a genuinely hard problem.
#
#  INDEPENDENT SIGNALS (not derived from Close):
#   • Fear_Greed   — crowd sentiment
#   • BTC_Dom      — market structure
#   • CSV_*        — rank, momentum from snapshot
#   • Vol_Ratio    — volume vs its own average
#   • HL_Range     — intraday range
# ──────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   symbol: str = "",
                   master_df: pd.DataFrame = None,
                   fear_greed_series: pd.Series = None,
                   btc_dom_series: pd.Series = None) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    close  = out['Close'].squeeze()
    high   = out['High'].squeeze()
    low    = out['Low'].squeeze()
    volume = out['Volume'].squeeze()

    # ── 1. Log Return — TARGET signal (col 0) ────────────
    out['Log_Return'] = np.log(close / close.shift(1)).squeeze()

    # ── 2. RSI (14) ───────────────────────────────────────
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi   = (100 - (100 / (1 + gain / loss))).squeeze()
    out['RSI'] = rsi

    # ── 3. Stochastic RSI ─────────────────────────────────
    out['Stoch_RSI'] = (
        (rsi - rsi.rolling(14).min()) /
        (rsi.rolling(14).max() - rsi.rolling(14).min() + 1e-10)
    ).squeeze()

    # ── 4. MACD Histogram only (removes redundancy) ───────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    out['MACD_Hist'] = (macd - macd.ewm(span=9, adjust=False).mean()).squeeze()

    # ── 5. Bollinger Width + %B (two signals, not three) ──
    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out['BB_Width'] = ((bb_upper - bb_lower) / (sma20 + 1e-10)).squeeze()
    out['BB_PctB']  = ((close - bb_lower) / (bb_upper - bb_lower + 1e-10)).squeeze()

    # ── 6. Price position vs SMA50/200 ────────────────────
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out['Price_vs_SMA50']  = ((close - sma50)  / (sma50  + 1e-10)).squeeze()
    out['Price_vs_SMA200'] = ((close - sma200) / (sma200 + 1e-10)).squeeze()

    # ── 7. EMA_50 (trend anchor) ──────────────────────────
    out['EMA_50'] = close.ewm(span=50, adjust=False).mean().squeeze()

    # ── 8. Volume ratio (relative, not absolute) ──────────
    vol_ma20 = volume.rolling(20).mean()
    out['Vol_Ratio']  = (volume / (vol_ma20 + 1e-10)).squeeze()
    out['OBV_Normed'] = (volume * np.sign(close.diff())).cumsum().squeeze()

    # ── 9. ATR (14) ───────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    out['ATR'] = tr.rolling(14).mean().squeeze()

    # ── 10. Williams %R ───────────────────────────────────
    hh = high.rolling(14).max()
    ll = low.rolling(14).min()
    out['Williams_R'] = (((hh - close) / (hh - ll + 1e-10)) * -100).squeeze()

    # ── 11. Multi-timeframe momentum ──────────────────────
    out['Return_3d']  = close.pct_change(3).squeeze()
    out['Return_7d']  = close.pct_change(7).squeeze()
    out['Return_14d'] = close.pct_change(14).squeeze()
    out['Return_30d'] = close.pct_change(30).squeeze()

    # ── 12. Intraday range ────────────────────────────────
    out['HL_Range'] = ((high - low) / (close + 1e-10)).squeeze()

    # ── 13. EXTERNAL: Fear & Greed ────────────────────────
    if fear_greed_series is not None and not fear_greed_series.empty:
        fg = fear_greed_series.reindex(out.index, method='ffill').fillna(0.5)
        out['Fear_Greed'] = fg.values
    else:
        out['Fear_Greed'] = 0.5

    # ── 14. EXTERNAL: BTC Dominance ───────────────────────
    if btc_dom_series is not None and not btc_dom_series.empty:
        dom = btc_dom_series.reindex(out.index, method='ffill').fillna(0.45)
        out['BTC_Dom'] = dom.values
    else:
        out['BTC_Dom'] = 0.45

    # ── 15. CSV CONTEXT (static broadcast) ────────────────
    if master_df is not None and symbol:
        ctx = get_csv_context(symbol, master_df)
    else:
        ctx = {"rank_norm": 0.5, "chg_24h": 0.0, "chg_7d": 0.0,
               "chg_30d": 0.0, "vol_rank_norm": 0.5}

    out['CSV_Rank']    = ctx["rank_norm"]
    out['CSV_Mom_24h'] = ctx["chg_24h"] / 100.0
    out['CSV_Mom_7d']  = ctx["chg_7d"]  / 100.0
    out['CSV_Mom_30d'] = ctx["chg_30d"] / 100.0
    out['CSV_VolRank'] = ctx["vol_rank_norm"]

    out.dropna(inplace=True)
    return out


# ──────────────────────────────────────────────
#  SEQUENCE BUILDER
#  FIXED:
#  y = next-day Log_Return (NOT Close)
# ──────────────────────────────────────────────
def make_sequences(scaled: np.ndarray,
                   lookback: int = 90,
                   target_idx: int = 0):
    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])

        # Predict next-day Log_Return
        y.append(scaled[i, target_idx])

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
#  BUILD MODEL
# ──────────────────────────────────────────────
def build_model(n_features: int, lookback: int = 90) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=(lookback, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8,  activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model


# ──────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ──────────────────────────────────────────────
def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    data  = data.squeeze()
    delta = data.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def get_signal(rsi_value: float) -> str:
    if rsi_value < 30: return "BUY"
    if rsi_value > 70: return "SELL"
    return "HOLD"


def get_price_changes(hist_df: pd.DataFrame) -> dict:
    current_p  = float(hist_df['Close'].iloc[-1].squeeze())
    prev_day   = float(hist_df['Close'].iloc[-2].squeeze())
    prev_week  = float(hist_df['Close'].iloc[-7].squeeze()) if len(hist_df) > 7 else current_p
    return {
        "current":    current_p,
        "change_24h": ((current_p - prev_day)  / prev_day)  * 100,
        "change_7d":  ((current_p - prev_week) / prev_week) * 100,
    }


# ──────────────────────────────────────────────
#  LSTM PREDICTION v4 — FIXED
#  REAL prediction:
#  predict Log_Return → convert to price
# ──────────────────────────────────────────────
def run_prediction(hist_df: pd.DataFrame,
                   symbol: str = "",
                   master_df: pd.DataFrame = None) -> dict:

    LOOKBACK = 90

    fear_greed = fetch_fear_greed(n_days=3000)
    btc_dom    = fetch_btc_dominance(pd.to_datetime(hist_df.index))

    featured = build_features(
        hist_df,
        symbol=symbol,
        master_df=master_df,
        fear_greed_series=fear_greed,
        btc_dom_series=btc_dom
    )

    n_features = featured.shape[1]

    # Find Log_Return column index
    cols = list(featured.columns)
    log_return_idx = cols.index('Log_Return')

    scaler = RobustScaler()
    scaled = scaler.fit_transform(featured.values)

    # Build sequences using Log_Return target
    X, y = make_sequences(
        scaled,
        LOOKBACK,
        target_idx=log_return_idx
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]

    model = build_model(n_features, LOOKBACK)

    model.fit(
        X,
        y,
        batch_size=32,
        epochs=60,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0
    )

    # ─────────────────────────────
    # Predict next-day Log_Return
    # ─────────────────────────────
    last_seq = scaled[-LOOKBACK:].reshape(
        1,
        LOOKBACK,
        n_features
    )

    pred_log_return_scaled = float(
        model.predict(last_seq, verbose=0)[0][0]
    )

    # Inverse transform ONLY Log_Return column
    dummy = np.zeros((1, n_features))
    dummy[0, log_return_idx] = pred_log_return_scaled

    pred_log_return = float(
        scaler.inverse_transform(dummy)[0][log_return_idx]
    )

    # ─────────────────────────────
    # Convert return → price
    # ─────────────────────────────
    last_close = float(featured['Close'].iloc[-1])

    pred_price = float(
        last_close * np.exp(pred_log_return)
    )

    pct_change = (
        (pred_price / last_close) - 1
    ) * 100

    return {
        "price":      pred_price,
        "pct_change": pct_change,
        "model":      model,
        "scaler":     scaler,
        "featured":   featured,
        "fear_greed": fear_greed,
        "btc_dom":    btc_dom,
    }


# ──────────────────────────────────────────────
#  FAST VALIDATION — zero retraining
# ──────────────────────────────────────────────
def run_validation(hist_df: pd.DataFrame,
                   cached_model=None,
                   cached_scaler=None,
                   cached_featured=None) -> dict:
    if hist_df is None or len(hist_df) < 30:
        return {
            "is_accurate": False,
            "mape": 0,
            "comp_df": pd.DataFrame(),
            "error": "Not enough data after history deletion/reset"
        }

    TEST_SIZE = 30
    LOOKBACK  = 90

    featured  = cached_featured if cached_featured is not None else build_features(hist_df)
    cols      = list(featured.columns)
    close_idx = cols.index('Close')

    close_col = featured['Close'].squeeze().values
    actuals   = close_col[-TEST_SIZE:]

    if cached_model is not None and cached_scaler is not None:
        n_features = featured.shape[1]
        scaled     = cached_scaler.transform(featured.values)
        preds      = []

        for i in range(TEST_SIZE):
            end_idx           = len(scaled) - TEST_SIZE + i
            seq_in            = scaled[end_idx - LOOKBACK:end_idx].reshape(1, LOOKBACK, n_features)
            pred_close_scaled = float(cached_model.predict(seq_in, verbose=0)[0][0])

            dummy                = np.zeros((1, n_features))
            dummy[0, close_idx]  = pred_close_scaled
            pred_price           = float(cached_scaler.inverse_transform(dummy)[0][close_idx])

            # Sanity clamp
            base_close = float(close_col[-(TEST_SIZE - i) - 1])
            pred_price = float(np.clip(pred_price,
                                       base_close * 0.50,
                                       base_close * 1.50))
            preds.append(pred_price)
    else:
        close_series = featured['Close'].squeeze()
        ema10        = close_series.ewm(span=10, adjust=False).mean().shift(1)
        preds        = ema10.values[-TEST_SIZE:].tolist()

    actuals = np.array(actuals, dtype=float)
    preds   = np.array(preds,   dtype=float)
    mask    = ~(np.isnan(actuals) | np.isnan(preds))
    actuals, preds = actuals[mask], preds[mask]

    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-10))) * 100)

    return {
        "mape":        mape,
        "comp_df":     pd.DataFrame({
            'Actual':    actuals.flatten(),
            'Predicted': preds.flatten(),
            'Deviation': np.abs(actuals - preds).flatten(),
        }),
        "is_accurate": mape < 5,
    }


# ──────────────────────────────────────────────
#  TOP MOVERS
# ──────────────────────────────────────────────
def get_top_gainers(master_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return master_df.nlargest(n, '24h')[['Coin Name', 'Symbol', '24h']]


def get_top_losers(master_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return master_df.nsmallest(n, '24h')[['Coin Name', 'Symbol', '24h']]


# ══════════════════════════════════════════════
#  ANALYSIS SUITE
# ══════════════════════════════════════════════

def get_ath_atl(hist_df: pd.DataFrame) -> dict:
    close    = hist_df['Close'].squeeze()
    ath, atl = float(close.max()), float(close.min())
    current  = float(close.iloc[-1])
    return {
        "ath": ath, "atl": atl,
        "ath_date": str(close.idxmax())[:10],
        "atl_date": str(close.idxmin())[:10],
        "pct_from_ath": ((current - ath) / ath) * 100,
        "pct_from_atl": ((current - atl) / atl) * 100,
        "current": current,
    }


def get_monthly_performance(hist_df: pd.DataFrame) -> pd.DataFrame:
    df = hist_df[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    df['Close'] = df['Close'].squeeze()
    monthly     = df['Close'].resample('ME').last()
    monthly_ret = (monthly.pct_change() * 100).dropna()
    result = pd.DataFrame({
        'Year':     monthly_ret.index.year,
        'Month':    monthly_ret.index.strftime('%b'),
        'MonthNum': monthly_ret.index.month,
        'Return':   monthly_ret.values.round(2),
    })
    def label(r):
        if r >= 20:  return "🚀 Very High"
        if r >= 5:   return "📈 High"
        if r >= 0:   return "🟢 Slight Gain"
        if r >= -5:  return "🟡 Slight Loss"
        if r >= -20: return "📉 Low"
        return "🔴 Very Low"
    result['Label'] = result['Return'].apply(label)
    return result.reset_index(drop=True)


def get_volatility_analysis(hist_df: pd.DataFrame) -> pd.DataFrame:
    close       = hist_df['Close'].squeeze()
    rolling_vol = close.pct_change().rolling(30).std() * np.sqrt(365) * 100
    vol_df = pd.DataFrame({'Date': rolling_vol.index,
                           'Volatility': rolling_vol.values}).dropna()
    vol_df['Date'] = pd.to_datetime(vol_df['Date'])
    return vol_df


def get_volume_anomalies(hist_df: pd.DataFrame) -> pd.DataFrame:
    df = hist_df.copy()
    df['Vol_MA30']  = df['Volume'].squeeze().rolling(30).mean()
    df['Vol_Ratio'] = df['Volume'].squeeze() / (df['Vol_MA30'] + 1e-10)
    close_s         = df['Close'].squeeze()
    df['Direction'] = np.where(
        close_s > close_s.shift(1), "📈 Bullish Surge", "📉 Bearish Dump"
    )
    return df[df['Vol_Ratio'] > 2.0][['Close', 'Volume', 'Vol_Ratio', 'Direction']].copy().tail(30)


def get_support_resistance(hist_df: pd.DataFrame, window: int = 20) -> dict:
    close   = hist_df['Close'].squeeze()
    current = float(close.iloc[-1])
    local_max = close[(close.shift(1) < close) & (close.shift(-1) < close)]
    local_min = close[(close.shift(1) > close) & (close.shift(-1) > close)]
    resistance = sorted([float(v) for v in local_max if v > current],
                        key=lambda x: abs(x - current))[:3]
    support    = sorted([float(v) for v in local_min if v < current],
                        key=lambda x: abs(x - current))[:3]
    return {"current": current, "resistance": resistance, "support": support}


def get_trend_analysis(hist_df: pd.DataFrame) -> dict:
    close   = hist_df['Close'].squeeze()
    ema200  = close.ewm(span=200, adjust=False).mean()
    ema50   = close.ewm(span=50,  adjust=False).mean()
    ema20   = close.ewm(span=20,  adjust=False).mean()
    current = float(close.iloc[-1])
    e200, e50, e20 = float(ema200.iloc[-1]), float(ema50.iloc[-1]), float(ema20.iloc[-1])
    if current > e200 and e50 > e200:
        trend, trend_color = "🐂 Bull Market", "green"
    elif current < e200 and e50 < e200:
        trend, trend_color = "🐻 Bear Market", "red"
    else:
        trend, trend_color = "↔ Sideways / Consolidation", "orange"
    streak = 0
    for val in reversed((close > ema200).values):
        if val == (close.iloc[-1] > ema200.iloc[-1]): streak += 1
        else: break
    return {
        "trend": trend, "trend_color": trend_color,
        "trend_strength": abs((e20 - e50) / (e50 + 1e-10)) * 100,
        "ema20": e20, "ema50": e50, "ema200": e200,
        "ema_streak_days": streak, "current": current,
    }


def get_price_zones(hist_df: pd.DataFrame) -> pd.DataFrame:
    close = hist_df['Close'].squeeze()
    p10, p25 = float(close.quantile(0.10)), float(close.quantile(0.25))
    p75, p90 = float(close.quantile(0.75)), float(close.quantile(0.90))
    def zone(v):
        if v >= p90: return "ATH Zone"
        if v >= p75: return "High"
        if v >= p25: return "Normal"
        if v >= p10: return "Low"
        return "ATL Zone"
    return pd.DataFrame({
        'Date':  pd.to_datetime(hist_df.index),
        'Close': close.values,
        'Zone':  close.apply(zone).values,
    })