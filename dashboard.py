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
    symbol = st.sidebar.text_input("Enter Stock Ticker", "AMZN").upper()

    # SPECIAL LOGIC FOR AMZN: Combine Historical CSV + Live Data
    if symbol == "AMZN":
        try:
            # 1. Load historical AMZN from CSV (2013-2019)
            hist_df = pd.read_csv('portfolio_data.csv')
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df.set_index('Date', inplace=True)
            historical_amzn = hist_df[['AMZN']].rename(columns={'AMZN': 'Close'})

            # 2. Fetch live AMZN data from the end of the CSV to Today
            live_amzn = yf.download(symbol, start='2019-12-01', end=datetime.now(), auto_adjust=True)
            if isinstance(live_amzn.columns, pd.MultiIndex):
                live_amzn.columns = live_amzn.columns.get_level_values(0)
            live_amzn = live_amzn[['Close']]

            # 3. Merge them
            df = pd.concat([historical_amzn, live_amzn])
            st.sidebar.success("Successfully Merged CSV + Live AMZN Data")
        except FileNotFoundError:
            st.sidebar.warning("CSV not found. Falling back to Live Data only.")
            df = yf.download(symbol, start='2015-01-01', end=datetime.now(), auto_adjust=True)
    else:
        # Standard Live Fetch for any other ticker
        df = yf.download(symbol, start='2015-01-01', end=datetime.now(), auto_adjust=True)

    # Cleanup for standard yfinance structure
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']]

else:
    try:
        # Load the raw portfolio_data.csv
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
    st.subheader(f"Data Overview: {symbol}")
    col_table, col_stats = st.columns([2, 1])
    with col_table:
        # Fixed the width warning
        st.dataframe(df.tail(10), width=1000)
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
                st.info("💡 **Analysis:** Trading below historical average. Potential buying opportunity.")
            else:
                st.info("💡 **Analysis:** Trading at a premium. Market sentiment is high.")

    # --- PREDICTION SECTION ---
    with col2:
        if st.button("🤖 Predict with LSTM"):
            st.markdown("### AI Prediction Results")
            with st.spinner("Training LSTM on Combined Data..."):

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

                current_val = float(df['Close'].iloc[-1])
                diff = pred_price[0][0] - current_val

                # 4. Display Results
                st.metric("Predicted Price (Next Day)", f"${pred_price[0][0]:.2f}", delta=f"{diff:.2f}")

                if diff > 0:
                    st.success("✅ **Recommendation:** Upward trend predicted. High potential for investment.")
                else:
                    st.warning("❌ **Recommendation:** Downward trend predicted. Suggest waiting for a dip.")

else:
    st.info("Select a data source from the sidebar to begin analysis.")
    ##& "H:\AI project\StockPredictionLSTM\.venv\Scripts\python.exe" -m streamlit run "H:\AI project\StockPredictionLSTM\dashboard.py"