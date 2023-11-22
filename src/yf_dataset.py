# Ingestion of data from APIs

import yfinance as yf


def get_stock_data(ticker, start_date, period):
    # Download stock data using yf.download
    data = yf.download(tickers=ticker, start=start_date, period=period)
    return data
