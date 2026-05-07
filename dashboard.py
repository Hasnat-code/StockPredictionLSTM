import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime
import os

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Crypto AI Master Pro",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>
    🪙 Crypto AI Master Pro: Analysis & Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------- SESSION STATE ----------------
if 'total_preds' not in st.session_state:
    st.session_state.total_preds = 0

if 'correct_preds' not in st.session_state:
    st.session_state.correct_preds = 0

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

# ---------------- FILES ----------------
users_file = "users.txt"

# ---------------- LOGIN FUNCTIONS ----------------
def save_user(username, password):

    if not os.path.exists(users_file):

        with open(users_file, "w") as f:
            f.write("Username|Password\n")

    with open(users_file, "a") as f:
        f.write(f"{username}|{password}\n")


def user_exists(username, password):

    if not os.path.exists(users_file):
        return False

    with open(users_file, "r") as f:

        lines = f.readlines()

        for line in lines:

            if "|" in line:

                parts = line.strip().split("|")

                if len(parts) == 2:

                    saved_user = parts[0]
                    saved_pass = parts[1]

                    if (
                        saved_user == username
                        and saved_pass == password
                    ):
                        return True

    return False

# ---------------- UTILITIES ----------------
def clean_val(x):

    if isinstance(x, str):

        clean_str = (
            x.replace('$', '')
            .replace(',', '')
            .replace('%', '')
            .strip()
        )

        if clean_str in ['', '-']:
            return np.nan

        return float(clean_str)

    return x


# ---------------- UPDATE ACTUAL VALUES ----------------
def update_actual_prices(username):

    file_path = (
        f"{username}_prediction_history.txt"
    )

    if not os.path.exists(file_path):
        return

    updated_lines = []

    with open(file_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        if "Actual: Pending" in line:

            try:

                parts = line.split("|")

                coin = parts[1].strip()

                ticker = f"{coin}-USD"

                latest_data = yf.download(
                    ticker,
                    period="1d",
                    progress=False
                )

                if not latest_data.empty:

                    actual_price = float(
                        latest_data['Close']
                        .iloc[-1]
                        .squeeze()
                    )

                    line = line.replace(
                        "Pending",
                        f"{actual_price:,.2f}"
                    )

            except:
                pass

        updated_lines.append(line)

    with open(file_path, "w") as f:

        f.writelines(updated_lines)


# ---------------- SAVE PREDICTIONS ----------------
def log_prediction(
        username,
        coin,
        pred,
        actual="Pending"
):

    file_path = (
        f"{username}_prediction_history.txt"
    )

    date_str = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    new_entry = (
        f"{date_str} | "
        f"{coin} | "
        f"Prediction: {pred:,.2f} | "
        f"Actual: {actual}\n"
    )

    if not os.path.exists(file_path):

        with open(file_path, "w") as f:

            f.write(
                f"Prediction History of {username}\n"
            )

            f.write("-" * 60 + "\n")

    with open(file_path, "a") as f:
        f.write(new_entry)


# ---------------- RSI ----------------
def calculate_rsi(data, window=14):

    delta = data.diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(window=window)
        .mean()
    )

    loss = (
        (-delta.where(delta < 0, 0))
        .rolling(window=window)
        .mean()
    )

    rs = gain / loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.subheader("🔐 Login / Register")

    login_tab, register_tab = st.tabs(
        ["Login", "Register"]
    )

    # -------- LOGIN --------
    with login_tab:

        login_user = st.text_input(
            "Username",
            key="login_user"
        )

        login_pass = st.text_input(
            "Password",
            type="password",
            key="login_pass"
        )

        if st.button("Login"):

            if user_exists(
                login_user,
                login_pass
            ):

                st.session_state.logged_in = True
                st.session_state.username = login_user

                update_actual_prices(login_user)

                st.success(
                    f"Welcome {login_user}"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password"
                )

    # -------- REGISTER --------
    with register_tab:

        reg_user = st.text_input(
            "Create Username",
            key="reg_user"
        )

        reg_pass = st.text_input(
            "Create Password",
            type="password",
            key="reg_pass"
        )

        if st.button("Register"):

            if (
                reg_user.strip() == ""
                or reg_pass.strip() == ""
            ):

                st.warning(
                    "Fill all fields"
                )

            else:

                save_user(
                    reg_user,
                    reg_pass
                )

                st.success(
                    "Account created successfully"
                )

    st.stop()

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_master():

    df = pd.read_csv("CryptocurrencyData.csv")

    df.columns = df.columns.str.strip()

    cols_to_fix = [
        'Price',
        '24h',
        '7d',
        '30d',
        'Market Cap'
    ]

    for col in cols_to_fix:

        if col in df.columns:
            df[col] = df[col].apply(clean_val)

    return df


master_df = load_master()

# ---------------- TOP MOVERS ----------------
st.subheader("🚀 Top Market Movers (24h)")

col1, col2 = st.columns(2)

with col1:

    st.write("### Top Gainers")

    st.dataframe(
        master_df.nlargest(5, '24h')[
            ['Coin Name', 'Symbol', '24h']
        ],
        hide_index=True
    )

with col2:

    st.write("### Top Losers")

    st.dataframe(
        master_df.nsmallest(5, '24h')[
            ['Coin Name', 'Symbol', '24h']
        ],
        hide_index=True
    )

# ---------------- SIDEBAR ----------------
st.sidebar.success(
    f"Logged in as: {st.session_state.username}"
)

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.username = ""

    st.rerun()

st.sidebar.header("🪙 Coin Directory")

search = st.sidebar.text_input(
    "Search Coin Name",
    ""
)

filtered = master_df[
    master_df['Coin Name'].str.contains(
        search,
        case=False,
        na=False
    )
]

selected_name = st.sidebar.selectbox(
    "Select Asset",
    filtered['Coin Name'].tolist()
)

selected_info = master_df[
    master_df['Coin Name'] == selected_name
].iloc[0]

symbol = str(selected_info['Symbol']).strip()

# ---------------- FETCH LIVE DATA ----------------
@st.cache_data
def fetch_live(coin_symbol):

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

    return data[['Close']]


hist_df = fetch_live(symbol)

# ---------------- MAIN DASHBOARD ----------------
if not hist_df.empty:

    st.write("---")

    st.subheader(
        f"📈 Price Change Indicator: {selected_name}"
    )

    ind1, ind2, ind3 = st.columns(3)

    current_p = float(
        hist_df['Close'].iloc[-1].squeeze()
    )

    prev_day = float(
        hist_df['Close'].iloc[-2].squeeze()
    )

    prev_week = (
        float(
            hist_df['Close'].iloc[-7].squeeze()
        )
        if len(hist_df) > 7
        else current_p
    )

    change_24h = (
        (current_p - prev_day)
        / prev_day
    ) * 100

    change_7d = (
        (current_p - prev_week)
        / prev_week
    ) * 100

    ind1.metric(
        "Current Price",
        f"${current_p:,.2f}"
    )

    ind2.metric(
        "24h Change",
        f"{change_24h:.2f}%",
        delta=f"{change_24h:.2f}%"
    )

    ind3.metric(
        "7d Change",
        f"{change_7d:.2f}%",
        delta=f"{change_7d:.2f}%"
    )

    # ---------------- CHART ----------------
    st.subheader("📊 Historical Price Chart")

    st.line_chart(hist_df['Close'])

    # ---------------- RSI ----------------
    rsi_vals = calculate_rsi(hist_df['Close'])

    latest_rsi = float(
        rsi_vals.iloc[-1].squeeze()
    )

    st.write("---")

    st.subheader("📡 Technical Signal")

    sig_col, acc_col = st.columns(2)

    with sig_col:

        if latest_rsi < 30:

            st.success(
                "🟢 BUY Signal "
                "(Oversold Market)"
            )

        elif latest_rsi > 70:

            st.error(
                "🔴 SELL Signal "
                "(Overbought Market)"
            )

        else:

            st.info(
                "🟡 HOLD Signal "
                "(Neutral Trend)"
            )

        st.write(
            f"Current RSI: {latest_rsi:.2f}"
        )

    with acc_col:

        st.write("### Accuracy Tracker")

        if st.session_state.total_preds > 0:

            acc_pct = (
                st.session_state.correct_preds
                / st.session_state.total_preds
            ) * 100

        else:
            acc_pct = 0

        st.write(
            f"Total Predictions: "
            f"{st.session_state.total_preds}"
        )

        st.write(
            f"Correct Predictions: "
            f"{st.session_state.correct_preds}"
        )

        st.progress(acc_pct / 100)

        st.write(
            f"Accuracy: {acc_pct:.2f}%"
        )

    st.write("---")

    # ---------------- BUTTONS ----------------
    col_predict, col_validate = st.columns(2)

    # -------- PREDICTION --------
    with col_predict:

        if st.button("🤖 Predict Tomorrow"):

            with st.spinner("Training AI Model..."):

                st.session_state.total_preds += 1

                scaler = MinMaxScaler(
                    feature_range=(0, 1)
                )

                scaled = scaler.fit_transform(
                    hist_df.values
                )

                x = []
                y = []

                for i in range(60, len(scaled)):

                    x.append(
                        scaled[i - 60:i, 0]
                    )

                    y.append(
                        scaled[i, 0]
                    )

                x = np.array(x)
                y = np.array(y)

                x = np.reshape(
                    x,
                    (x.shape[0], x.shape[1], 1)
                )

                model = Sequential()

                model.add(
                    LSTM(
                        128,
                        return_sequences=True,
                        input_shape=(60, 1)
                    )
                )

                model.add(Dropout(0.2))

                model.add(
                    LSTM(
                        64,
                        return_sequences=False
                    )
                )

                model.add(Dense(1))

                model.compile(
                    optimizer='adam',
                    loss='mean_squared_error'
                )

                model.fit(
                    x,
                    y,
                    batch_size=1,
                    epochs=5,
                    verbose=0
                )

                last_60 = scaled[-60:]

                last_60 = np.reshape(
                    last_60,
                    (1, 60, 1)
                )

                pred_raw = model.predict(
                    last_60,
                    verbose=0
                )

                pred = scaler.inverse_transform(
                    pred_raw
                )[0][0]

                diff = pred - current_p

                st.metric(
                    "AI Predicted Price",
                    f"${pred:,.2f}",
                    delta=f"{diff:,.2f}"
                )

                log_prediction(
                    st.session_state.username,
                    symbol,
                    pred
                )

                st.success(
                    "Prediction saved successfully."
                )

    # -------- VALIDATION --------
    with col_validate:

        if st.button("📊 Validate Accuracy"):

            st.subheader("Backtesting Report")

            dataset = hist_df.values

            scaler = MinMaxScaler(
                feature_range=(0, 1)
            )

            scaled = scaler.fit_transform(
                dataset
            )

            test_size = 15

            train_data = scaled[:-test_size]

            xt = []
            yt = []

            for i in range(60, len(train_data)):

                xt.append(
                    train_data[i - 60:i, 0]
                )

                yt.append(
                    train_data[i, 0]
                )

            xt = np.array(xt)
            yt = np.array(yt)

            xt = np.reshape(
                xt,
                (xt.shape[0], xt.shape[1], 1)
            )

            model_dev = Sequential()

            model_dev.add(
                LSTM(
                    64,
                    input_shape=(60, 1)
                )
            )

            model_dev.add(Dense(1))

            model_dev.compile(
                optimizer='adam',
                loss='mean_squared_error'
            )

            model_dev.fit(
                xt,
                yt,
                batch_size=1,
                epochs=3,
                verbose=0
            )

            inputs = scaled[
                len(scaled) - test_size - 60:
            ]

            x_test = []

            for i in range(60, 60 + test_size):

                x_test.append(
                    inputs[i - 60:i, 0]
                )

            x_test = np.array(x_test)

            x_test = np.reshape(
                x_test,
                (
                    x_test.shape[0],
                    x_test.shape[1],
                    1
                )
            )

            preds = scaler.inverse_transform(
                model_dev.predict(
                    x_test,
                    verbose=0
                )
            )

            actuals = dataset[-test_size:]

            mape = np.mean(
                np.abs(
                    (actuals - preds)
                    / actuals
                )
            ) * 100

            st.metric(
                "Average Deviation",
                f"{mape:.2f}%"
            )

            if mape < 5:
                st.session_state.correct_preds += 1

            comp_df = pd.DataFrame({
                'Actual': actuals.flatten(),
                'Predicted': preds.flatten()
            })

            st.line_chart(comp_df)

else:

    st.warning(
        "⚠ No valid market data found."
    )

# ---------------- HISTORY ----------------
user_history_file = (
    f"{st.session_state.username}"
    f"_prediction_history.txt"
)

if os.path.exists(user_history_file):

    st.write("---")

    st.subheader("📜 Your Prediction History")

    with open(user_history_file, "r") as f:

        st.text(f.read())