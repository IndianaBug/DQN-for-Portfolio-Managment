import math
from pandas import DataFrame
import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import codecs
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

def logarithmic_epsilon_decay(episodes, max_epsilon, min_epsilon):
    """
    Logarithmic epsilon decay function.

    Args:
        episodes: The number of episodes.

    Returns:
        The epsilon value for a given episode number.
    """
    # Initialize epsilon with initial value
    epsilon = max_epsilon

    for episode in range(episodes):
        if episode == 0:
            yield 1
        # Calculate the decay factor for the current episode
        decay_factor = np.log10(epsilon / min_epsilon) / (episodes - 1)

        # Update epsilon with the decay factor
        epsilon = epsilon * np.power(10.0, -decay_factor)

        # Yield the updated epsilon value
        yield epsilon

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

def get_dataset(mode='raw', horizon=5, number_lags=100):

    # Load Coca's data
    df = pd.DataFrame(yf.Ticker("KO").history(start='2018-01-01', end='2023-09-30', interval='1d'))
    # Add 10, 20, 50, 100 moving averages, both simple and exponential
    for length in [5, 10, 20, 50, 100, 200, 365]:
        df[f'ma_{length}'] = df['Close'].rolling(length).mean()
        #df[f'ma_{length}'] = ((df[f'ma_{length}'] - df['Close']) / df['Close']) * 100
        df[f'ema_{length}'] = df['Close'].ewm(span = length, adjust = False ).mean()
        #df[f'ema_{length}'] = ((df[f'ema_{length}'] - df['Close']) / df['Close']) * 100
    df['rsi'] = RSI(df["Close"], 14)
    # Add bollinger bands
    df['upper_band']= bollinger_bands(df["Close"])['2']
    #df['upper_band'] = ((df['upper_band'] - df['Close']) / df['Close']) * 100
    df['mid_band']= bollinger_bands(df["Close"])['0']
    #df['mid_band']= ((df['mid_band'] - df['Close']) / df['Close']) * 100
    df['lower_band']= bollinger_bands(df["Close"])['-2']
    #df['lower_band']= ((df['lower_band'] - df['Close']) / df['Close']) * 100
    # Add OBV
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    #for i in [2, 4, 8, 16, 32, 64]:
        #df[f'OBV_diff_{i}'] = df['obv'].diff(i)
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
    df.columns = df.columns.str.lower()
    # Remove columns
    df = df.drop(['dividends', 'stock splits', 'open', 'high', 'low',
                'day_of_year', 'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'log_ret', 'return'], axis=1)

    scale_max = float(np.amax(df['close'].values))
    scale_min = float(np.amin(df['close'].values))
    
    if mode == 'raw':
        return df
    
        # Length of the trainign dataset
    TRAIN_SPLIT = math.ceil(len(df.close)*0.9*0.9)
    read_df = pd.DataFrame()
    read_df.index = pd.to_datetime(df.index)

    # Scale prices
    scaler = MinMaxScaler()
    scaler.fit(df['close'].values[:TRAIN_SPLIT].reshape(-1, 1))
    read_df['close'] = scaler.transform(df['close'].values.reshape(-1, 1)).reshape(-1)

    # Add lags
    for period in range(1, number_lags+1):
      read_df[f'lag_{period}'] = read_df['close'].shift(periods = period, axis = 0)

    # Add multy variate data
    multy_variate_Closing_Price = []
    for index in range(len(df)):
      if index > horizon-1:
        multy_variate_Closing_Price.append((read_df.close.values[index-horizon:index]))

    # Add to dataframe
    read_df = read_df[horizon:]
    read_df['multy_close'] = multy_variate_Closing_Price

    if mode=='eval':
        read_df = read_df.drop(['close'], axis=1)
        read_df = read_df.dropna()

        # Add indicators
        cols = ['volume', 'ma_10', 'ema_10', 'ma_20', 'ema_20', 'ma_50',
                'ema_50', 'ma_100', 'ema_100', 'rsi', 'upper_band', 'mid_band',
                'lower_band', 
                # 'obv_diff_2', 'obv_diff_4', 'obv_diff_8', 'obv_diff_16','obv_diff_32', 'obv_diff_64', 
                'william', 'macd', 'signal', 'k_fast',
                'd_fast', 'k_slow', 'd_slow', 'volatility', 'slope_2', 'slope_4',
                'slope_6', 'slope_12', 'slope_24', 'slope_48', 'day_of_year_sin',
                'day_of_week_sin', 'day_of_month_sin', 'week_of_year_sin', 'month_sin',
                'quarter_sin']
        
        indicators = df[cols]
        indicators = indicators.loc[indicators.index >= read_df.index[0]]
        scaler_indicators = MinMaxScaler()
        scaler_indicators.fit(indicators)
        read_df[indicators.columns] = scaler_indicators.transform(indicators)

        # Move target column to the beggining
        column_to_move = read_df.pop('multy_close')
        read_df.insert(0, 'multy_close', column_to_move)

        columns_to_shift = read_df.columns[1:]
        read_df[columns_to_shift] = read_df[columns_to_shift].shift(horizon)

        read_df = read_df.dropna()
        read_df = read_df.tail(1200)

        return read_df

    if mode == 'predict':
        read_df = read_df.dropna()
        read_df = read_df.drop('multy_close', axis=1)


        # Add indicators
        cols = ['volume', 'ma_10', 'ema_10', 'ma_20', 'ema_20', 'ma_50',
                'ema_50', 'ma_100', 'ema_100', 'rsi', 'upper_band', 'mid_band',
                'lower_band', 
                #'obv_diff_2', 'obv_diff_4', 'obv_diff_8', 'obv_diff_16',
                #'obv_diff_32', 'obv_diff_64', 
                'william', 'macd', 'signal', 'k_fast',
                'd_fast', 'k_slow', 'd_slow', 'volatility', 'slope_2', 'slope_4',
                'slope_6', 'slope_12', 'slope_24', 'slope_48', 'day_of_year_sin',
                'day_of_week_sin', 'day_of_month_sin', 'week_of_year_sin', 'month_sin',
                'quarter_sin']
        
        indicators = df[cols]
        indicators = indicators.loc[indicators.index >= read_df.index[0]]
        close = read_df['close'].values
        scaler_indicators = MinMaxScaler()

        scaler_Close = MinMaxScaler()

        scaler_Close = scaler_Close.fit(close.reshape(1, -1))


        scaler_indicators.fit(indicators)

        read_df[indicators.columns] = scaler_indicators.transform(indicators)
        read_df[read_df.columns] = read_df[read_df.columns].shift(horizon)
        read_df = read_df.dropna()
        read_df = read_df.tail(1200)


        return read_df, scale_max, scale_min



def mean_absolute_percentage_error(y_true, y_pred):
    """
      Error calculator
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_train_history(history, title):
  """
    Helper for neural nets
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.grid()
  plt.show()