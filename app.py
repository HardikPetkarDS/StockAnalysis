import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import os
import datetime

st.set_page_config(page_title="Stock Price Forecast (LSTM)", layout="wide")
st.title("📈 Stock Market Trend Forecasting — LSTM Model")
st.write("Enter a stock ticker to see historical data, visualizations, and next-day prediction using an LSTM model.")

# -----------------------
# Sidebar Inputs
# -----------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", "AAPL")
    lookback = st.slider("Lookback Days", 30, 120, 60)
    start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())
    submit = st.button("🔍 Load Data")

# -----------------------
# Helper Functions
# -----------------------

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    return df

def create_sequences(values, lookback):
    X = []
    for i in range(len(values) - lookback):
        X.append(values[i:i+lookback])
    return np.array(X)

# -----------------------
# Main Execution
# -----------------------

if submit:
    st.subheader(f"📌 Stock Data for {ticker.upper()}")
    df = download_data(ticker, start_date, end_date)

    if df is None:
        st.error("⚠️ No data found. Check ticker symbol.")
        st.stop()

    st.dataframe(df.tail())

    # Plot Close Price
    st.subheader("📉 Close Price History")
    st.line_chart(df["Close"])

    # Check if model exists
    model_path = f"models/lstm_{ticker}.h5"
    scaler_path = f"models/scaler_{ticker}.pkl"

    model_available = os.path.exists(model_path) and os.path.exists(scaler_path)

    # Prepare data
    close_data = df[['Close']].values

    scaler = MinMaxScaler()
    
    if model_available:
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(close_data)
    else:
        scaled = scaler.fit_transform(close_data)

    if len(scaled) < lookback + 1:
        st.error("⚠️ Not enough data for the chosen lookback period.")
        st.stop()

    X_all = create_sequences(scaled, lookback)
    last_window = np.expand_dims(X_all[-1], axis=0)

    # -----------------------
    # Prediction
    # -----------------------
    st.subheader("📊 LSTM Prediction")

    if model_available:
        try:
            model = load_model(model_path)
            pred_scaled = model.predict(last_window)
            pred = scaler.inverse_transform(pred_scaled)[0][0]

            # Show USD prediction
            st.success(f"📌 **Predicted Next-Day Close Price: {pred:.2f} USD**")

            # Convert USD → INR
            import requests
            try:
                rate = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()['rates']['INR']
                pred_in_inr = pred * rate
                st.metric("Predicted Price (INR)", f"{pred_in_inr:.2f} ₹")
            except:
                st.warning("Could not convert USD to INR.")

        except:
            st.error("Error loading the trained model.")
    else:
        st.warning("⚠️ No trained model found. Please upload model files in /models/ folder.")
        st.info("Fallback: Showing last closing price instead of prediction.")
        st.metric("Last Close Price", f"{df['Close'].iloc[-1]:.2f} USD")

    # -----------------------
    # Actual vs Predicted Chart
    # -----------------------
    results_csv = f"models/results_{ticker}.csv"

    if os.path.exists(results_csv):
        st.subheader("📈 Actual vs Predicted (Test Set)")

        results = pd.read_csv(results_csv)
        results["date"] = df.index[-len(results):]

        chart_df = results.melt(id_vars="date", value_vars=["y_true", "y_pred"], var_name="Type", value_name="Price")

        chart = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x="date:T",
                y="Price:Q",
                color="Type:N"
            )
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        st.info("No saved predictions file found. (results_TICKER.csv)")
