import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Crypto AI Master + Backtester", layout="wide")
st.markdown("<h1 style='text-align: center;'>🪙 Crypto Market Analysis & LSTM Prediction</h1>", unsafe_allow_html=True)


# --- 2. DATA UTILITIES ---
def clean_val(x):
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '').replace('%', '').strip()
        return np.nan if clean_str in ['-', ''] else float(clean_str)
    return x


@st.cache_data
def load_crypto_master():
    df = pd.read_csv('CryptocurrencyData.csv')
    df.columns = df.columns.str.strip()
    cols_to_fix = ['Price', '24h', '7d', '30d', 'Market Cap']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(clean_val)
    return df


master_df = load_crypto_master()

# --- 3. SIDEBAR SELECTION ---
st.sidebar.header("Explore 4,000+ Coins")
search = st.sidebar.text_input("Search Coin Name", "")
filtered = master_df[master_df['Coin Name'].str.contains(search, case=False)]

selected_name = st.sidebar.selectbox("Select Coin", filtered['Coin Name'].tolist())
selected_info = master_df[master_df['Coin Name'] == selected_name].iloc[0]
symbol = str(selected_info['Symbol']).strip()


# --- 4. FETCH LIVE DATA ---
@st.cache_data
def fetch_history(coin_symbol):
    ticker = f"{coin_symbol}-USD"
    data = yf.download(ticker, start="2020-01-01", end=datetime.now(), auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data[['Close']]
    return pd.DataFrame()


hist_df = fetch_history(symbol)

# --- 5. DASHBOARD MAIN VIEW ---
if not hist_df.empty:
    st.subheader(f"Snapshot Analysis: {selected_name} ({symbol})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Global Rank", f"#{selected_info['Rank']}")
    m2.metric("Snapshot Price", f"${selected_info['Price']:,.2f}")
    m3.metric("24h Change", f"{selected_info['24h']}%", delta=f"{selected_info['24h']}%")
    m4.metric("Market Cap", f"${selected_info['Market Cap']:,.0f}")

    st.subheader(f"Live Price Action (2020 - 2026)")
    st.line_chart(hist_df['Close'])

    # --- ACTION BUTTONS (Original + New) ---
    col_analyze, col_predict, col_dev = st.columns(3)

    # A. PREVIOUS FEATURE: ANALYZE
    with col_analyze:
        if st.button("📊 Analyze Volatility"):
            st.write("### Trend Analysis")
            st.write(f"**7-Day Performance:** {selected_info['7d']}%")
            st.write(f"**30-Day Performance:** {selected_info['30d']}%")
            if abs(selected_info['24h']) > 10:
                st.error("⚠️ Sudden High Volatility detected.")
            else:
                st.info("✅ Stable short-term movement.")

    # B. PREVIOUS FEATURE: PREDICT
    with col_predict:
        if st.button("🤖 Run LSTM Prediction"):
            st.subheader("AI Prediction & Verdict")
            with st.spinner("Processing deep learning sequences..."):
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(hist_df.values)

                x, y = [], []
                for i in range(60, len(scaled)):
                    x.append(scaled[i - 60:i, 0])
                    y.append(scaled[i, 0])
                x, y = np.array(x), np.array(y)
                x = np.reshape(x, (x.shape[0], x.shape[1], 1))

                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape=(60, 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x, y, batch_size=1, epochs=5, verbose=0)

                last_60 = scaled[-60:]
                pred = model.predict(np.reshape(last_60, (1, 60, 1)))
                final_pred = scaler.inverse_transform(pred)[0][0]
                current = hist_df['Close'].iloc[-1]
                diff = final_pred - current

                st.metric("Predicted Price (Tomorrow)", f"${final_pred:,.2f}", delta=f"{diff:,.2f}")
                if diff > 0:
                    st.success("🎯 **VERDICT: INVEST**")
                else:
                    st.warning("📉 **VERDICT: HOLD**")

    # C. NEW FEATURE: CHECK DEVIATION
    with col_dev:
        if st.button("📉 Check Model Deviation"):
            st.subheader("Backtest & Accuracy Report")
            with st.spinner("Calculating error margins..."):
                # Use live data but hide the last 15 days
                test_window = 15
                dataset = hist_df.values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(dataset)

                train_len = len(dataset) - test_window
                train_data = scaled[0:train_len, :]

                xt, yt = [], []
                for i in range(60, len(train_data)):
                    xt.append(train_data[i - 60:i, 0])
                    yt.append(train_data[i, 0])

                # Fast model for deviation check
                model_dev = Sequential()
                model_dev.add(LSTM(64, return_sequences=False, input_shape=(60, 1)))
                model_dev.add(Dense(1))
                model_dev.compile(optimizer='adam', loss='mean_squared_error')
                model_dev.fit(np.array(xt), np.array(yt), batch_size=1, epochs=3, verbose=0)

                # Predict the 'Hidden' 15 days
                inputs = scaled[train_len - 60:]
                x_test = []
                for i in range(60, 60 + test_window):
                    x_test.append(inputs[i - 60:i, 0])

                preds = model_dev.predict(np.array(x_test))
                preds = scaler.inverse_transform(preds)
                actuals = dataset[train_len:]

                # Calculate Error
                mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
                st.metric("Average Deviation", f"{mape:.2f}%")
                st.progress(max(0, min(100, int(100 - mape))))

                # Deviation Graph
                comparison = pd.DataFrame({
                    'Actual': actuals.flatten(),
                    'Predicted': preds.flatten()
                })
                st.write("**Actual vs Predicted (Last 15 Days)**")
                st.line_chart(comparison)

else:
    st.error("Select a coin with valid market history to begin.")