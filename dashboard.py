import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
st.markdown("<h1 style='text-align: center;'>Stock Market Analysis 📈 + Prediction using LSTM</h1>",
            unsafe_allow_html=True)

# --- 2. DATA LOADING LOGIC ---
st.sidebar.header("Control Panel")
data_source = st.sidebar.radio("Select Data Source", ("Live Stocks", "Portfolio Dataset (CSV)"))

df = pd.DataFrame()
symbol = ""

if data_source == "Live Stocks":
    symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
    df = yf.download(symbol, start=start_date, end=datetime.now(), auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df = df[['Close']]

else:
    try:
        # Load your specific portfolio_data.csv
        df_raw = pd.read_csv('portfolio_data.csv')
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw.set_index('Date', inplace=True)

        symbol = st.sidebar.selectbox("Select Asset from CSV", df_raw.columns)
        df = df_raw[[symbol]].rename(columns={symbol: 'Close'})
        st.sidebar.success(f"Loaded {symbol} from CSV")
    except FileNotFoundError:
        st.error("portfolio_data.csv not found! Please place it in the project folder.")

# --- 3. DASHBOARD MAIN VIEW ---
if not df.empty:
    # Row 1: Data Table & Stats
    st.subheader(f"Historical Data Overview: {symbol}")
    col_table, col_stats = st.columns([2, 1])
    with col_table:
        st.dataframe(df.tail(10), use_container_width=True)
    with col_stats:
        st.write("**Quick Stats**")
        st.write(df.describe())

    # Row 2: Interactive Chart
    st.subheader("Price Visualization (XY Plane)")
    st.line_chart(df['Close'])

    # Row 3: Action Buttons
    col1, col2 = st.columns(2)

    # --- ANALYZE SECTION ---
    with col1:
        if st.button("📊 Analyze Trend"):
            st.markdown("### Trend Analysis Report")

            # Monthly Analysis
            df_monthly = df['Close'].resample('ME').last().pct_change()
            best_month = df_monthly.idxmax()
            worst_month = df_monthly.idxmin()

            # Sudden Highs/Lows
            st.metric("All-Time High", f"${df['Close'].max():.2f}")
            st.metric("All-Time Low", f"${df['Close'].min():.2f}")

            st.success(f"🚀 **Sudden High:** Peak growth occurred in **{best_month.strftime('%B %Y')}**")
            st.error(f"⚠️ **Sudden Low:** Sharpest drop occurred in **{worst_month.strftime('%B %Y')}**")

            # Suggestion Logic
            current_price = df['Close'].iloc[-1]
            avg_price = df['Close'].mean()
            if current_price < avg_price:
                st.info(
                    "💡 **Analysis:** Stock is currently trading below its historical average. This might be a dip worth investigating.")
            else:
                st.info("💡 **Analysis:** Stock is trading at a premium compared to its historical average.")

    # --- PREDICTION SECTION ---
    with col2:
        if st.button("🤖 Predict with LSTM"):
            st.markdown("### AI Prediction Results")
            with st.spinner("Training LSTM Model... Please wait (this takes ~60s)"):

                # 1. Preprocess
                dataset = df.values
                training_data_len = int(np.ceil(len(dataset) * .95))
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)

                train_data = scaled_data[0:int(training_data_len), :]
                x_train, y_train = [], []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i - 60:i, 0])
                    y_train.append(train_data[i, 0])

                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                # 2. Build & Train
                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

                # 3. Predict Tomorrow
                last_60_days = scaled_data[-60:]
                x_input = np.array([last_60_days])
                x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

                pred_price = model.predict(x_input)
                pred_price = scaler.inverse_transform(pred_price)

                current_val = df['Close'].iloc[-1]
                diff = pred_price[0][0] - current_val

                # 4. Display Results
                st.metric("Predicted Price (Next Day)", f"${pred_price[0][0]:.2f}", delta=f"{diff:.2f}")

                if diff > 0:
                    st.success(
                        "✅ **Recommendation:** The model predicts an upward trend. High potential for investment.")
                else:
                    st.warning(
                        "❌ **Recommendation:** The model predicts a downward move. Caution advised before investing.")

else:
    st.info("Select a data source from the sidebar to begin analysis.")