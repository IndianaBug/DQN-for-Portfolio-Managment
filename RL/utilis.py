import math
from pandas import DataFrame
import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import codecs

def time_features(dt, timecycle):
    if timecycle == 'day':
        a = math.sin(2 * math.pi * dt / 7.)
    if timecycle == 'month':
        a = math.sin(2 * math.pi * dt / 30.)
    return a

def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)
def bollinger_bands(series: pd.Series, length: int = 20, *, num_stds: tuple[float, ...] = (2, 0, -2), prefix: str = '') -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/74283044/
    rolling = series.rolling(length)
    bband0 = rolling.mean()
    bband_std = rolling.std(ddof=0)
    return pd.DataFrame({f'{prefix}{num_std}': (bband0 + (bband_std * num_std)) for num_std in num_stds})
def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr
def MACD(
        cls,
        ohlc: DataFrame,
        period_fast: int = 12,
        period_slow: int = 26,
        signal: int = 9,
        column: str = "Close",
        adjust: bool = True,
    ) -> DataFrame:
    """
    MACD, MACD Signal and MACD difference.
    The MACD Line oscillates above and below the zero line, which is also known as the centerline.
    These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course, depends on the direction of the moving average cross.
    Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as the shorter EMA diverges further from the longer EMA.
    This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA is below the 26-day EMA.
    Negative values increase as the shorter EMA diverges further below the longer EMA. This means downside momentum is increasing.
    Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.
    As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.
    A bullish crossover occurs when the MACD turns up and crosses above the signal line.
    A bearish crossover occurs when the MACD turns down and crosses below the signal line.
    """
    EMA_fast = pd.Series(
        ohlc[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
        name="EMA_fast",
    )
    EMA_slow = pd.Series(
        ohlc[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
        name="EMA_slow",
    )
    MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
    MACD_signal = pd.Series(
        MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
    )
    return pd.concat([MACD, MACD_signal], axis=1)

def stochastics(dataframe, low, high, close, k, d ):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal
    When the %K crosses below %D, sell signal
    """
    df = dataframe.copy()
    # Set minimum low and maximum high of the k stoch
    low_min  = df[low].rolling( window = k ).min()
    high_max = df[high].rolling( window = k ).max()
    # Fast Stochastic
    df['k_fast'] = 100 * (df[close] - low_min)/(high_max - low_min)
    df['k_fast'].ffill(inplace=True)
    df['d_fast'] = df['k_fast'].rolling(window = d).mean()
    # Slow Stochastic
    df['k_slow'] = df["d_fast"]
    df['d_slow'] = df['k_slow'].rolling(window = d).mean()
    return df

def get_dataset():
    # Load Coca's data
    df = pd.DataFrame(yf.Ticker("KO").history(start='2018-01-01', end='2023-09-30', interval='1d'))
    # Add 10, 20, 50, 100 moving averages, both simple and exponential
    for length in [10, 20, 50, 100]:
        df[f'ma_{length}'] = df['Close'].rolling(length).mean()
        df[f'ma_{length}'] = ((df[f'ma_{length}'] - df['Close']) / df['Close']) * 100
        df[f'ema_{length}'] = df['Close'].ewm(span = length, adjust = False ).mean()
        df[f'ema_{length}'] = ((df[f'ema_{length}'] - df['Close']) / df['Close']) * 100
    df['rsi'] = RSI(df["Close"], 14)
    # Add bollinger bands
    df['upper_band']= bollinger_bands(df["Close"])['2']
    df['upper_band'] = ((df['upper_band'] - df['Close']) / df['Close']) * 100
    df['mid_band']= bollinger_bands(df["Close"])['0']
    df['mid_band']= ((df['mid_band'] - df['Close']) / df['Close']) * 100
    df['lower_band']= bollinger_bands(df["Close"])['-2']
    df['lower_band']= ((df['lower_band'] - df['Close']) / df['Close']) * 100
    # Add OBV
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    for i in [2, 4, 8, 16, 32, 64]:
        df[f'OBV_diff_{i}'] = df['obv'].diff(i)
    # Add william r%
    df['william'] = get_wr(df['High'], df['Low'], df['Close'], 14)
    # Add MACD
    df['macd'] = MACD(df, df)['MACD']
    df['signal'] = MACD(df, df)['SIGNAL']
    # Add stochastic
    df = stochastics(df, 'Low', 'High', 'Close', 14, 3 )
    # 7 day volatility
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=7).std() * np.sqrt(7)
    # Trend (Slope)
    for ran in [2, 4, 6, 12, 24, 48]:
        ar = []
        for i in range(len(df)):
            try:
                close_1 = df['Close'].iloc[i]
                close_2 = df['Close'].iloc[i-ran]
                slope = (close_1 - close_2) / ran
            except:
                slope = np.nan
            ar.append(slope)
        df[f'Slope_{ran}'] = np.array(ar)
    # Add time cycles
    df['Day_of_Year'] = df.index.dayofyear
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Week_of_Year'] = df.index.isocalendar().week
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    cycles = ['Day_of_Year', 'Day_of_Week', 'Day_of_Month', 'Week_of_Year', 'Month', 'Quarter']
    for cycle in cycles:
        df[f'{cycle}_sin'] = np.sin(2 * np.pi * df[cycle] / df[cycle].max())
    # Add returns
    df.insert(0, 'return', df['Open'].pct_change())
    df_shifted = df.drop('return', axis=1)
    df_shifted = df_shifted.shift(1) # Shift data backwards
    df_shifted.insert(0, 'return', df['return'])
    df_shifted = df_shifted[:-1] # Remove the first line
    df = df_shifted.tail(1200)
    df.columns = df.columns.str.lower()
    # Remove columns
    df = df.drop(['dividends', 'stock splits', 'close', 'open', 'high', 'low',
                'day_of_year', 'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'obv', 'log_ret'], axis=1)
    return df

def load_data_structure(file):
    return json.load(codecs.open(file, 'r', encoding='utf-8'))

def save_data_structure(file_path, key, value):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}
    data[key] = value
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def plot_loss(training_losses, validation_losses):
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs with Early Stopping')
    plt.legend()
    plt.show()