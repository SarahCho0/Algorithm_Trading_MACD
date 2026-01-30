import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ==========================================
# 1. 핵심 함수 정의
# ==========================================

def calc_ma_recursive(series, n, alpha):
    data = series.values
    length = len(data)
    result = np.full(length, np.nan)

    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return pd.Series(result, index=series.index)

    first_valid_idx = np.argmax(valid_mask)

    if length < first_valid_idx + n:
        return pd.Series(result, index=series.index)

    k_factor = (1 - alpha) / (1 - alpha**n)
    alpha_n = alpha**n

    initial_window = data[first_valid_idx : first_valid_idx + n]
    weights = np.array([alpha**i for i in range(n)])
    w_sum = weights.sum()
    normalized_weights = (weights / w_sum)[::-1]

    start_idx = first_valid_idx + n - 1
    result[start_idx] = np.dot(initial_window, normalized_weights)

    for t in range(start_idx + 1, length):
        prev_s = result[t-1]
        curr_x = data[t]
        old_x = data[t-n]

        if np.isnan(curr_x) or np.isnan(old_x):
            result[t] = np.nan
            continue

        result[t] = alpha * prev_s + k_factor * (curr_x - alpha_n * old_x)

    return pd.Series(result, index=series.index)

def calculate_performance(change_series, macd_series, signal_series):
    position = np.where(macd_series > signal_series, 1, 0)
    position_series = pd.Series(position, index=change_series.index).shift(1)
    position_series = position_series.fillna(0)

    strategy_returns = position_series * change_series
    cum_returns = (1 + strategy_returns).cumprod()

    if len(cum_returns) == 0:
        return 0, pd.Series(dtype=float), 0

    final_return = cum_returns.iloc[-1] - 1

    # 승률 계산을 위한 매매 횟수 체크 (출력용은 아님)
    trades = position_series.diff().abs().sum() / 2

    if trades > 0:
        trade_ids = (position_series != position_series.shift()).cumsum()
        holding_mask = position_series == 1
        if holding_mask.sum() > 0:
            per_trade_returns = strategy_returns[holding_mask].groupby(trade_ids[holding_mask]).apply(lambda x: (1 + x).prod() - 1)
            win_count = (per_trade_returns > 0).sum()
            win_rate = win_count / len(per_trade_returns)
        else:
            win_rate = 0
    else:
        win_rate = 0

    return final_return, cum_returns, win_rate

# ==========================================
# 2. 데이터 준비 및 기간 분리
# ==========================================
TICKER = '000660' #005930, ks200
START_DATE = '2015-01-01'
END_DATE = '2025-12-31'

print(f"[{TICKER}] 데이터 로딩")
df_all = fdr.DataReader(TICKER, START_DATE, END_DATE)

if 'Change' not in df_all.columns:
    df_all['Change'] = df_all['Close'].pct_change()

train_df = df_all.loc['2015-01-01':'2019-12-31'].copy()
test_df = df_all.loc['2020-01-01':'2025-12-31'].copy()

print(f"Train 기간: {train_df.index[0].date()} ~ {train_df.index[-1].date()}")
print(f"Test  기간: {test_df.index[0].date()} ~ {test_df.index[-1].date()}")

# ==========================================
# 3. [Train] 파라미터 최적화
# ==========================================

FAST_RANGE = range(5, 25, 5)
SLOW_RANGE = range(20, 60, 10)
SIGNAL_RANGE = range(5, 15, 2)
ALPHA_RANGE = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

best_train_result = {'score': -np.inf, 'params': None}
total_iter = len(ALPHA_RANGE) * len(FAST_RANGE) * len(SLOW_RANGE) * len(SIGNAL_RANGE)
pbar = tqdm(total=total_iter)

train_close = train_df['Close']
train_change = train_df['Change']

for alpha in ALPHA_RANGE:
    for fast in FAST_RANGE:
        ma_fast = calc_ma_recursive(train_close, fast, alpha)
        for slow in SLOW_RANGE:
            if fast >= slow:
                pbar.update(len(SIGNAL_RANGE))
                continue
            ma_slow = calc_ma_recursive(train_close, slow, alpha)
            macd_full = ma_fast - ma_slow
            for signal in SIGNAL_RANGE:
                signal_full = calc_ma_recursive(macd_full, signal, alpha)
                ret, _, _ = calculate_performance(train_change, macd_full, signal_full)
                if ret > best_train_result['score']:
                    best_train_result['score'] = ret
                    best_train_result['params'] = (alpha, fast, slow, signal)
                pbar.update(1)
pbar.close()

if best_train_result['params'] is None:
    print("최적화 실패: 유효한 결과를 찾지 못했습니다.")
else:
    b_alpha, b_fast, b_slow, b_sig = best_train_result['params']
    print(f"   Alpha: {b_alpha}")
    print(f"   Fast: {b_fast} / Slow: {b_slow} / Signal: {b_sig}")
    print(f"   Train 수익률: {best_train_result['score']*100:.2f}%")

    # ==========================================
    # 4. [Test] 실전 백테스트
    # ==========================================
    print("\n" + "="*60)
    print("검증 데이터 실전 테스트")
    print("="*60)

    test_buy_hold = (1 + test_df['Change'].fillna(0)).cumprod()

    # 표준 MACD
    std_ema12 = df_all['Close'].ewm(span=12, adjust=False).mean()
    std_ema26 = df_all['Close'].ewm(span=26, adjust=False).mean()
    std_macd = std_ema12 - std_ema26
    std_sig = std_macd.ewm(span=9, adjust=False).mean()

    std_ret, std_curve, _ = calculate_performance(
        df_all.loc[test_df.index, 'Change'],
        std_macd.loc[test_df.index],
        std_sig.loc[test_df.index]
    )

    # 최적화 모델
    final_ma_fast = calc_ma_recursive(df_all['Close'], b_fast, b_alpha)
    final_ma_slow = calc_ma_recursive(df_all['Close'], b_slow, b_alpha)
    final_macd = final_ma_fast - final_ma_slow
    final_signal = calc_ma_recursive(final_macd, b_sig, b_alpha)

    oos_ret, oos_curve, _ = calculate_performance(
        df_all.loc[test_df.index, 'Change'],
        final_macd.loc[test_df.index],
        final_signal.loc[test_df.index]
    )

    print(f"최종 성적표 (2020.01 ~ 2025.12)")
    print(f"{'구분':<15} | {'수익률':<10}")
    print("-" * 35)
    print(f"{'표준 MACD':<15} | {std_ret*100:6.2f}%")
    print(f"{'최적화 모델':<15}| {oos_ret*100:6.2f}%")
    print("-" * 35)

    # 그래프 그리기
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    t_macd = final_macd.loc[train_df.index]
    t_sig = final_signal.loc[train_df.index]
    _, t_curve, _ = calculate_performance(train_change, t_macd, t_sig)
    t_bh = (1 + train_change.fillna(0)).cumprod()

    plt.plot(t_curve.index, t_curve, label='Optimized', color='red')
    plt.plot(t_bh.index, t_bh, label='Buy&Hold', color='gray', linestyle='--')
    plt.title(f'[TRAIN] Return: {(t_curve.iloc[-1]-1)*100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(oos_curve.index, oos_curve, label='Optimized (OOS)', color='blue')
    plt.plot(test_buy_hold.index, test_buy_hold, label='Buy&Hold', color='gray', linestyle='--')
    plt.title(f'[TEST] Return: {oos_ret*100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
