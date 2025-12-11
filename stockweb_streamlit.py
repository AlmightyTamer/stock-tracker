# stock_tracker_app.py
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

st.title("📈 Simple Stock Tracker")

# Input box for ticker
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, TSLA):")

if ticker:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo")  # last 1 month
    
    if not data.empty:
        # Show current price
        price = round(data['Close'][-1], 2)
        st.write(f"**{ticker.upper()} current price:** ${price}")
        
        # Show line chart
        st.line_chart(data['Close'])
    else:
        st.error("Invalid ticker symbol!")
