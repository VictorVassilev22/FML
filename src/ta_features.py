# Library for data modeling and creating features

import pandas as pd
import pandas_ta as tapd
import numpy as np
from matplotlib import pyplot as plt


def add_pct_ch_and_future(df: pd.DataFrame, period=5):
    # Create 5-day % changes of Adj_Close for the current day, and n days in the future
    df[str(period) + 'd_close_pct'] = df['Adj_Close'].pct_change(period) #calc current % change

    df[str(period) + 'd_future_close'] = df['Adj_Close'].shift(-period) # calc future close price
    df[str(period) + 'd_close_future_pct'] = df[str(period) + 'd_future_close'].pct_change(period) #calc future % change
    features = df.columns
    return features


def add_sma_normalized(df: pd.DataFrame, periods=None):
    # Create moving averages for time periods of 14, 30, 50, and 200
    if periods is None:
        periods = [14, 30, 50, 200]

    for n in periods:
        # Create the moving average indicator and divide by Adj_Close
        df['sma' + str(n)] = tapd.sma(df['Adj_Close'],
                                     length=n) / df['Adj_Close']
    return df.columns

def add_sma(df: pd.DataFrame, periods=None):
    # Create moving averages for time periods of 14, 30, 50, and 200
    if periods is None:
        periods = [14, 30, 50, 200]

    for n in periods:
        # Create the moving average indicator
        df['sma' + str(n)] = tapd.sma(df['Adj_Close'], length=n)
    return df.columns

def add_ema(df: pd.DataFrame, periods=None):
    # Create moving averages for time periods of 12, 26, 50, and 200

    #Short-Term Trends:
    # Period: 12 to 26 periods.
    # Why: Shorter periods make EMA more responsive to recent price changes, making it suitable for capturing short-term trends.

    # Medium-Term Trends:
    # Period: 50 periods.
    # Why: A 50-period EMA is often used to capture medium-term trends and filter out shorter-term noise.

    # Long-Term Trends:
    # Period: 200 periods.
    # Why: The 200-period EMA is commonly used for long-term trend analysis, particularly in financial markets.
    if periods is None:
        periods = [12, 26, 50, 200]

    for n in periods:
        # Create the moving average indicator
        df['ema' + str(n)] = tapd.ema(df['Adj_Close'], length=n)
    return df.columns


def add_wma(df: pd.DataFrame, period=5, custom_weights=5):

     # If custom_weights is not provided, use equal weights
    if custom_weights is None:
        custom_weights = np.arange(1, period + 1)

    # Check if the length of custom_weights matches the specified period
    if len(custom_weights) != period:
        raise ValueError("Length of custom_weights should be equal to wma_period.")

    # Calculate WMA with pandas_ta
    df['wma' + str(period)] = tapd.wma(df['Adj_Close'], length=period, weights=custom_weights)

    return df.columns

def add_rsi(df: pd.DataFrame, periods=None):
    # Create rsi for time periods of 14, 30, 50, and 200
    if periods is None:
        periods = [14, 30, 50, 200]

    for n in periods:
        # Create the RSI indicator
        df['rsi' + str(n)] = tapd.rsi(df['Adj_Close'], length=n)

    return df.columns


def add_sma_rsi_sma_x_rsi(df, periods=None):
    if periods is None:
        periods = [14, 30, 50, 200]

    for n in periods:
        # Create the moving average indicator and divide by Adj_Close
        df['sma' + str(n)] = tapd.sma(df['Adj_Close'],
                                     length=n) / df['Adj_Close'] #Why here is normalized?

        # Create the RSI indicator
        df['rsi' + str(n)] = tapd.rsi(df['Adj_Close'], length=n)

        # Create non-linear interaction SMA * RSI
        df['SMAxRSI_' + str(n)] = df['sma' + str(n)] * df['rsi' + str(n)]

    return df.columns


def get_most_correlated_feature(corr, target_column):
    """
    Function to find the most correlated feature to the target column.

    Args:
        corr (pandas.DataFrame): The correlation matrix.
        target_column (str): The name of the target column.

    Returns:
        str: The name of the most correlated feature.
    """
    # Get correlations between the target column and all other columns
    target_correlations = corr[target_column].drop(target_column)

    # Find the feature with the highest absolute correlation to the target column
    most_corr_feature = target_correlations.abs().idxmax()

    return most_corr_feature


def get_n_most_correlated_features(corr, target_column, n=1):
    """
    Function to find the n most correlated features to the target column.

    Args:
        corr (pandas.DataFrame): The correlation matrix.
        target_column (str): The name of the target column.
        n (int): The number of most correlated features to return. Default is 1.

    Returns:
        list: A list with the names of the n most correlated features, sorted by their absolute correlation values.
    """
    # Get correlations between the target column and all other columns
    target_correlations = corr[target_column].drop(target_column)

    # Find the n features with the highest absolute correlation to the target column
    most_corr_features = target_correlations.abs().nlargest(n).index.tolist()

    return most_corr_features


def add_volume_1d_pct_change_sma_plot_chart_hist(df, sma_period_change=5):
    # Create 2 new volume features: 1-day % volume change and n-day SMA of the % change
    df['Volume_1d_change'] = df['Volume'].pct_change()
    df['Volume_' + str(sma_period_change) + 'd_change_SMA'] = \
        tapd.sma(df['Volume_1d_change'], length=sma_period_change)

    # Plot histogram of volume % change SMA data
    df['Volume_' + str(sma_period_change) + 'd_change_SMA'].plot(kind='hist', sharex=False, bins=50)
    # Set labels
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title('Volume ' + str(sma_period_change) + ' day % change distribution')
    plt.show()
    return df.columns


def add_volume_1d_pct_change_sma(df, sma_period_change=5):
    # Create 2 new volume features, 1-day % volume change and n-day SMA of the % change
    df['Volume_1d_change'] = df['Volume'].pct_change()
    df['Volume_' + str(sma_period_change) + 'd_change_SMA'] = \
        tapd.sma(df['Volume_1d_change'], length=sma_period_change)

    return df.columns


def add_datetime_features(df, time_scales):
    # Check if the DataFrame has a 'Date' column or 'Date' as index
    if 'Date' not in df.columns and df.index.name != 'Date':
        raise ValueError("DataFrame must contain a 'Date' column or index")

    df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to pandas datetime objects

    for time_scale in time_scales:
        # Set the prefix for the get_dummies() function
        prefix = time_scale.lower()

        # Extract the desired time feature based on the time_scale input and validate the time_scale input
        if prefix == 'weekday':
            time_feature = df.index.dayofweek
        elif prefix == 'month':
            time_feature = df.index.month
        elif prefix == 'quarter':
            time_feature = df.index.quarter
        elif prefix == 'year':
            time_feature = df.index.year
        else:
            raise ValueError("time_scale must be one of the following: 'weekday', 'month', 'quarter', 'year'")

        # Create dummy variables for the specified time feature
        time_dummies = pd.get_dummies(time_feature, prefix=prefix, drop_first=True)

        # Rename columns to use more descriptive names
        time_dummies.columns = [f'{prefix}_{i}' for i in range(1, len(time_dummies.columns) + 1)]

        # Set the index of the time_dummies DataFrame to match the original DataFrame
        time_dummies.index = df.index

        # Join the dataframe with the days of week dataframe
        df = pd.concat([df, time_dummies], axis=1)

    # return all columns in the modified DataFrame
    return df.columns
