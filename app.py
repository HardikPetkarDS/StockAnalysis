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
import requests

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Stock Price Forecast (LSTM)", layout="wide")
st.title("📈 Stock Market Trend Forecasting — LSTM Model")
st.write("Enter a stock ticker to see historical data and next-day prediction using a pre-trained LSTM model.")

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", "AAPL")  # e.g. AAPL, RELIANCE.NS
    lookback = st.slider("Lookback Days (same as training)", 30, 120, 60)
    start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())
    submit = st.button("🔍 Load & Predict")

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    return df

def create_last_window(scaled_values, lookback):
    """Take the last `lookback` points and reshape for LSTM."""
    if len(scaled_values) < lookback:
        return None
    window = scaled_values[-lookback:]
    window = np.reshape(window, (1, lookback, 1))
    return window

def load_fx_rate_usd_inr():
    """Fetch live USD → INR rate. If it fails, return None."""
    try:
        resp = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        data = resp.json()
        return data["rates"]["INR"]
    except Exception:
        return None

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if submit:
    st.subheader(f"📌 Stock Data for {ticker.upper()}")

    # 1) Download data
    df = download_data(ticker, start_date, end_date)
    if df is None:
        st.error("⚠️ No data found. Check ticker symbol or date range.")
        st.stop()

    st.write("Latest rows of downloaded data:")
    st.dataframe(df.tail())

    # Plot Close Price
    st.subheader("📉 Close Price History")
    st.line_chart(df["Close"])

    # 2) Check for model + scaler
    model_path = f"models/lstm_{ticker}.h5"
    scaler_path = f"models/scaler_{ticker}.pkl"

    model_available = os.path.exists(model_path) and os.path.exists(scaler_path)

    if not model_available:
        st.warning("⚠️ No trained model found for this ticker in the /models folder.")
        st.info(f"Expected: lstm_{ticker}.h5 and scaler_{ticker}.pkl in models/")
        st.metric("Last Close Price", f"{df['Close'].iloc[-1]:.2f}")
        st.stop()

    # 3) Prepare data using SAME scaler as training
    close_data = df[["Close"]].values  # shape (N,1)

    try:
        scaler: MinMaxScaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

    # Transform using loaded scaler (do NOT fit again)
    try:
        scaled = scaler.transform(close_data)
    except Exception as e:
        st.error(f"Error scaling data. Make sure scaler matches this ticker. Details: {e}")
        st.stop()

    if len(scaled) <= lookback:
        st.error("⚠️ Not enough data points for the selected lookback.")
        st.stop()

    last_window = create_last_window(scaled, lookback)
    if last_window is None:
        st.error("⚠️ Could not create input window for LSTM.")
        st.stop()

    # 4) Load model and predict
    st.subheader("📊 LSTM Prediction")
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    try:
        pred_scaled = model.predict(last_window)
        # Inverse transform back to price
        pred = scaler.inverse_transform(pred_scaled)[0][0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Show USD/Original currency prediction
    st.success(f"📌 Predicted Next-Day Close Price: **{pred:.2f}** (model units, e.g. USD for AAPL)")

    # 5) Optional: Convert to INR assuming original is USD (like AAPL)
    st.markdown("#### 💱 Conversion to INR (assuming original price is in USD)")

    fx_rate = load_fx_rate_usd_inr()
    if fx_rate is not None:
        pred_in_inr = pred * fx_rate
        st.metric("Predicted Next-Day Close (INR)", f"{pred_in_inr:.2f} ₹")
        st.caption(f"Using live FX rate: 1 USD ≈ {fx_rate:.2f} INR")
    else:
        st.warning("Could not fetch live USD → INR rate. Please check your internet or API limit.")

    # 6) Actual vs Predicted from saved results (if available)
    st.subheader("📈 Actual vs Predicted (from test set, if available)")
    results_csv = f"models/results_{ticker}.csv"

    if os.path.exists(results_csv):
        try:
            results = pd.read_csv(results_csv)

            # Handle both possible column name styles:
            if {"Actual", "Predicted"}.issubset(results.columns):
                # from our training script
                chart_df = results.rename(columns={"Actual": "Actual", "Predicted": "Predicted"})
            elif {"y_true", "y_pred"}.issubset(results.columns):
                # from earlier version
                chart_df = results.rename(columns={"y_true": "Actual", "y_pred": "Predicted"})
            else:
                st.info("Results file found, but columns are not in expected format.")
                st.write("Expected columns: 'Actual'/'Predicted' or 'y_true'/'y_pred'.")
                st.dataframe(results.head())
                st.stop()

            # Create an index column for plotting
            chart_df["Index"] = np.arange(len(chart_df))

            melted = chart_df.melt(id_vars="Index", value_vars=["Actual", "Predicted"],
                                   var_name="Series", value_name="Price")

            chart = (
                alt.Chart(melted)
                .mark_line()
                .encode(
                    x="Index:Q",
                    y="Price:Q",
                    color="Series:N"
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not load or plot results CSV: {e}")
    else:
        st.info("No saved predictions file found (results_<TICKER>.csv). This is optional.")
