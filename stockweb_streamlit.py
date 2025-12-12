# stockweb_streamlit_personal.py
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# --- Title ---
st.title("📈 Param's Stock Tracker")

# Optional: Add a small description
st.write("Check current stock prices")

# --- Input box for ticker ---
ticker = st.text_input("Enter stock:")

if ticker:
    stock = yf.Ticker(ticker)
    data_day = stock.history(period="1d")  # for current price

    if not data_day.empty:
        price = round(data_day["Close"][-1], 2)
        st.write(f"**{ticker.upper()} current price:** ${price}")
    else:
        st.error("Invalid ticker symbol!")

    # --- Choose time period for graph ---
    st.write("Choose time period for graph:")
    period = st.selectbox("Select period:", ["1d", "5d", "1mo", "6mo", "1y", "max"])

    # Get historical data
    data = stock.history(period=period)

    if not data.empty:
        # Plot with matplotlib
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(data.index, data["Close"], marker='o')
        ax.set_title(f"{ticker.upper()} Price Chart ({period})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("No data available for this period")

# --- Footer ---
st.markdown("---")
st.markdown("Created by **Param Tyagi**")
