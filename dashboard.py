# ============================================================
#  dashboard.py  —  Backend: ALL logic, zero UI
#  Import this from frontend.py
#
#  CHANGES vs original:
#   1. fetch_live      → OHLCV + flattens yfinance MultiIndex columns (BUG FIX)
#   2. build_features  → squeeze() on every Series before assignment (BUG FIX)
#   3. make_sequences  → NEW: multi-feature sequence builder
#   4. build_model     → NEW: Bidirectional LSTM 128→64→32 (was 32→16)
#   5. run_prediction  → 50 epochs + EarlyStopping + 16 features (was 2 epochs, 1 feature)
#   6. run_validation  → no data leakage + 30 epochs + EarlyStopping (was 1 epoch, leaked)
# ============================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ──────────────────────────────────────────────
#  FILES
# ──────────────────────────────────────────────
USERS_FILE = "users.txt"


# ──────────────────────────────────────────────
#  AUTH  (unchanged)
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
#  UTILITIES  (unchanged)
# ──────────────────────────────────────────────
def clean_val(x):
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '').replace('%', '').strip()
        if clean_str in ['', '-']:
            return np.nan
        return float(clean_str)
    return x


# ──────────────────────────────────────────────
#  PREDICTION HISTORY  (unchanged)
# ──────────────────────────────────────────────
def log_prediction(username: str, coin: str, pred: float, actual: str = "Pending") -> None:
    file_path = f"{username}_prediction_history.txt"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = f"{date_str} | {coin} | Prediction: {pred:,.2f} | Actual: {actual}\n"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(f"Prediction History of {username}\n")
            f.write("-" * 60 + "\n")
    with open(file_path, "a") as f:
        f.write(new_entry)


def update_actual_prices(username: str) -> None:
    file_path = f"{username}_prediction_history.txt"
    if not os.path.exists(file_path):
        return
    updated_lines = deque()
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Actual: Pending" in line:
            try:
                parts = line.split("|")
                coin = parts[1].strip()
                ticker = f"{coin}-USD"
                latest_data = yf.download(ticker, period="1d", progress=False)
                if not latest_data.empty:
                    actual_price = float(latest_data['Close'].iloc[-1].squeeze())
                    line = line.replace("Pending", f"{actual_price:,.2f}")
            except:
                pass
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
#  DATA LOADING  (unchanged)
# ──────────────────────────────────────────────
def load_master() -> pd.DataFrame:
    df = pd.read_csv("CryptocurrencyData.csv")
    df.columns = df.columns.str.strip()
    for col in ['Price', '24h', '7d', '30d', 'Market Cap']:
        if col in df.columns:
            df[col] = df[col].apply(clean_val)
    return df


# ──────────────────────────────────────────────
#  FETCH LIVE  — IMPROVED + BUG FIXED
#
#  WHY THE ERROR HAPPENED:
#  Newer versions of yfinance return MultiIndex columns like:
#    ('Open','BTC-USD'), ('Close','BTC-USD'), ...
#  When you do arithmetic on those (e.g. bb_upper - bb_lower),
#  pandas produces a DataFrame with multiple columns instead of
#  a Series. Assigning that DataFrame to a single column key
#  raises: "Cannot set a DataFrame with multiple columns to
#  the single column BB_Width"
#
#  FIX: flatten MultiIndex to plain names right after download,
#  then squeeze() every column to guarantee 1-D Series.
# ──────────────────────────────────────────────
def fetch_live(coin_symbol: str) -> pd.DataFrame:
    ticker = f"{coin_symbol}-USD"
    data = yf.download(
        ticker,
        start="2020-01-01",
        end=datetime.now(),
        auto_adjust=True,
        progress=False
    )
    if data.empty:
        return pd.DataFrame()

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Flatten MultiIndex columns  e.g. ('Close','BTC-USD') → 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Squeeze any column that is still a 1-column DataFrame → plain Series
    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()

    return df


# ──────────────────────────────────────────────
#  FEATURE ENGINEERING  — NEW + BUG FIXED
#
#  Every intermediate value is .squeeze()'d before being
#  assigned to the output DataFrame.  This is the second
#  layer of defence against the MultiIndex / DataFrame
#  assignment error regardless of pandas version.
# ──────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : OHLCV dataframe (5 cols) from fetch_live()
    Output: 16-column enriched dataframe, NaN rows dropped
    Extra features:
      RSI(14), MACD, MACD_Signal, MACD_Hist,
      BB_Upper, BB_Lower, BB_Width,
      EMA_10, EMA_50, Vol_SMA, Return_1d, Return_5d
    """
    out = df.copy()

    # Force base columns to plain 1-D Series
    close  = out['Close'].squeeze()
    volume = out['Volume'].squeeze()

    # ── RSI (14) ──────────────────────────────
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    out['RSI'] = (100 - (100 / (1 + rs))).squeeze()

    # ── MACD (12 / 26 / 9) ───────────────────
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd        = (ema12 - ema26).squeeze()
    macd_signal = macd.ewm(span=9, adjust=False).mean().squeeze()
    macd_hist   = (macd - macd_signal).squeeze()
    out['MACD']        = macd
    out['MACD_Signal'] = macd_signal
    out['MACD_Hist']   = macd_hist

    # ── Bollinger Bands (20) ─────────────────
    sma20    = close.rolling(20).mean().squeeze()
    std20    = close.rolling(20).std().squeeze()
    bb_upper = (sma20 + 2 * std20).squeeze()
    bb_lower = (sma20 - 2 * std20).squeeze()
    bb_width = ((bb_upper - bb_lower) / sma20).squeeze()
    out['BB_Upper'] = bb_upper
    out['BB_Lower'] = bb_lower
    out['BB_Width'] = bb_width          # ← was the crash line; now safe

    # ── EMAs ─────────────────────────────────
    out['EMA_10'] = close.ewm(span=10, adjust=False).mean().squeeze()
    out['EMA_50'] = close.ewm(span=50, adjust=False).mean().squeeze()

    # ── Volume SMA (10) ──────────────────────
    out['Vol_SMA'] = volume.rolling(10).mean().squeeze()

    # ── Price Momentum ───────────────────────
    out['Return_1d'] = close.pct_change(1).squeeze()
    out['Return_5d'] = close.pct_change(5).squeeze()

    out.dropna(inplace=True)
    return out


# ──────────────────────────────────────────────
#  SEQUENCE BUILDER  — NEW
#  X : all 16 features over lookback window
#  y : next-day Close (column index 3 in scaled array)
# ──────────────────────────────────────────────
def make_sequences(scaled: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 3])             # 3 = Close column
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
#  BUILD MODEL  — IMPROVED
#  Was: LSTM 32→16 (too small for volatile crypto)
#  Now: Bidirectional LSTM 128→64→32 with dropout
# ──────────────────────────────────────────────
def build_model(n_features: int) -> Sequential:
    model = Sequential([
        Bidirectional(
            LSTM(128, return_sequences=True),
            input_shape=(60, n_features)
        ),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ──────────────────────────────────────────────
#  TECHNICAL INDICATORS  (unchanged logic)
#  Added .squeeze() on input for safety
# ──────────────────────────────────────────────
def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    data  = data.squeeze()
    delta = data.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def get_signal(rsi_value: float) -> str:
    """Returns 'BUY', 'SELL', or 'HOLD'"""
    if rsi_value < 30:
        return "BUY"
    elif rsi_value > 70:
        return "SELL"
    return "HOLD"


def get_price_changes(hist_df: pd.DataFrame) -> dict:
    current_p  = float(hist_df['Close'].iloc[-1].squeeze())
    prev_day   = float(hist_df['Close'].iloc[-2].squeeze())
    prev_week  = float(hist_df['Close'].iloc[-7].squeeze()) if len(hist_df) > 7 else current_p
    change_24h = ((current_p - prev_day)  / prev_day)  * 100
    change_7d  = ((current_p - prev_week) / prev_week) * 100
    return {
        "current":    current_p,
        "change_24h": change_24h,
        "change_7d":  change_7d,
    }


# ──────────────────────────────────────────────
#  LSTM PREDICTION  — IMPROVED
#  Was: 2 epochs, 1 feature, no val split
#  Now: 50 epochs + EarlyStopping + 16 features + val split + ReduceLR
# ──────────────────────────────────────────────
def run_prediction(hist_df: pd.DataFrame) -> float:
    """
    Trains improved LSTM and returns predicted next-day Close price.
    Stops automatically when val_loss stops improving (usually 15-30 epochs).
    """
    LOOKBACK = 60

    featured   = build_features(hist_df)
    n_features = featured.shape[1]          # 16

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(featured.values)

    X, y = make_sequences(scaled, LOOKBACK)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0
        )
    ]

    model = build_model(n_features)
    model.fit(
        X, y,
        batch_size=32,
        epochs=50,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0
    )

    last_seq    = np.reshape(scaled[-LOOKBACK:], (1, LOOKBACK, n_features))
    pred_scaled = model.predict(last_seq, verbose=0)[0][0]

    # Inverse-transform: put predicted scaled value into Close column (index 3)
    dummy       = np.zeros((1, n_features))
    dummy[0, 3] = pred_scaled
    pred_price  = scaler.inverse_transform(dummy)[0][3]

    return float(pred_price)


# ──────────────────────────────────────────────
#  LSTM BACKTESTING / VALIDATION  — IMPROVED
#  Was: 1 epoch, scaler on full data (leakage), 15-day test
#  Now: 30 epochs + EarlyStopping, train-only scaler, 20-day test
#  Returns same dict shape → frontend.py needs zero changes
# ──────────────────────────────────────────────
def run_validation(hist_df: pd.DataFrame) -> dict:
    """
    Backtests on last 20 days (was 15).
    Returns: { mape, comp_df, is_accurate }
    is_accurate is True when MAPE < 5% (same threshold as original)
    """
    LOOKBACK  = 60
    TEST_SIZE = 20

    featured   = build_features(hist_df)
    n_features = featured.shape[1]
    dataset    = featured.values

    # Strict split — scaler ONLY sees training rows
    train_raw = dataset[:-TEST_SIZE]
    test_raw  = dataset[-(TEST_SIZE + LOOKBACK):]   # needs 60 extra rows for sequences

    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)        # transform only — no fit

    X_train, y_train = make_sequences(train_scaled, LOOKBACK)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
    ]

    model = build_model(n_features)
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=30,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0
    )

    X_test = np.array([
        test_scaled[i - LOOKBACK:i]
        for i in range(LOOKBACK, LOOKBACK + TEST_SIZE)
    ])

    preds_scaled = model.predict(X_test, verbose=0)

    # Inverse-transform predictions (Close = col index 3)
    dummy_preds       = np.zeros((TEST_SIZE, n_features))
    dummy_preds[:, 3] = preds_scaled[:, 0]
    preds             = scaler.inverse_transform(dummy_preds)[:, 3]

    # Inverse-transform actuals
    dummy_act       = np.zeros((TEST_SIZE, n_features))
    dummy_act[:, 3] = test_scaled[LOOKBACK:, 3]
    actuals         = scaler.inverse_transform(dummy_act)[:, 3]

    mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

    comp_df = pd.DataFrame({
        'Actual':    actuals.flatten(),
        'Predicted': preds.flatten()
    })

    return {
        "mape":        mape,
        "comp_df":     comp_df,
        "is_accurate": mape < 5
    }


# ──────────────────────────────────────────────
#  TOP MOVERS  (unchanged)
# ──────────────────────────────────────────────
def get_top_gainers(master_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return master_df.nlargest(n, '24h')[['Coin Name', 'Symbol', '24h']]


def get_top_losers(master_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return master_df.nsmallest(n, '24h')[['Coin Name', 'Symbol', '24h']]