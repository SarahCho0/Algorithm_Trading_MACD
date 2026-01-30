import numpy as np
import pandas as pd

def calc_ma_recursive(series, n, alpha):
    data = series.values
    length = len(data)
    result = np.full(length, np.nan)
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask): return pd.Series(result, index=series.index)
    
    first_valid_idx = np.argmax(valid_mask)
    if length < first_valid_idx + n: return pd.Series(result, index=series.index)

    k_factor = (1 - alpha) / (1 - alpha**n)
    alpha_n = alpha**n
    initial_window = data[first_valid_idx : first_valid_idx + n]
    weights = np.array([alpha**i for i in range(n)])
    normalized_weights = (weights / weights.sum())[::-1]

    start_idx = first_valid_idx + n - 1
    result[start_idx] = np.dot(initial_window, normalized_weights)

    for t in range(start_idx + 1, length):
        if np.isnan(data[t]) or np.isnan(data[t-n]):
            result[t] = np.nan
            continue
        result[t] = alpha * result[t-1] + k_factor * (data[t] - alpha_n * data[t-n])
    
    return pd.Series(result, index=series.index)

def calculate_performance(change_series, macd_series, signal_series):
    position = (macd_series > signal_series).astype(int).shift(1).fillna(0)
    strategy_returns = position * change_series
    cum_returns = (1 + strategy_returns).cumprod()
    
    final_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0
    trades = position.diff().abs().sum() / 2
    
    win_rate = 0
    if trades > 0:
        trade_ids = (position != position.shift()).cumsum()
        holding = position == 1
        if holding.sum() > 0:
            per_trade = strategy_returns[holding].groupby(trade_ids[holding]).apply(lambda x: (1 + x).prod() - 1)
            win_rate = (per_trade > 0).mean()
            
    return final_return, cum_returns, win_rate