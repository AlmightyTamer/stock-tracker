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

import yfinance as yf
import matplotlib.pyplot as plt

def check_stock(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")

    if data.empty:
        print("Invalid ticker symbol!")
        return

    price = round(data["Close"][0], 2)
    print(f"{ticker.upper()} current price: ${price}")

def show_graph(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.empty:
        print("No data available for graph!")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"])
    plt.title(f"{ticker.upper()} Price Chart ({period})")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.show()

while True:
    symbol = input("Enter stock ticker (or 'quit' to exit): ")

    if symbol.lower() == "quit":
        break

    check_stock(symbol)

    # Ask if they want a graph
    choice = input("See graph? (y/n): ")
    if choice.lower() == "y":
        print("Choose time period:")
        print("1 = 1 day")
        print("5 = 5 days")
        print("1mo = 1 month")
        print("6mo = 6 months")
        print("1y = 1 year")
        print("max = max data")

        period = input("Enter period: ")
        show_graph(symbol, period)
# --- Footer ---
st.markdown("---")
st.markdown("Created by **Param Tyagi**")
