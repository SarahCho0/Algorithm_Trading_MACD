import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ==========================================
# 1. 핵심 함수 정의
# ==========================================

def calc_custom_ma(series, n, alpha):
    """
    지수 가중치와 유사한 사용자 정의 가중 이동평균
    Alpha가 클수록 과거 데이터 비중이 높음
    """
    weights = np.array([alpha**i for i in range(n)])
    w_sum = weights.sum()
    normalized_weights = (weights / w_sum)[::-1]
    return series.rolling(window=n).apply(lambda x: np.dot(x, normalized_weights), raw=True)

def calculate_performance(change_series, macd_series, signal_series):
    # 포지션: MACD > Signal (1: 보유, 0: 현금)
    position = np.where(macd_series > signal_series, 1, 0)

    # 수익률 적용 (신호 발생 다음날 진입 가정 -> shift(1))
    position_series = pd.Series(position, index=change_series.index).shift(1)
    position_series = position_series.fillna(0)

    strategy_returns = position_series * change_series

    # 누적 수익률 계산
    cum_returns = (1 + strategy_returns).cumprod()

    if len(cum_returns) == 0:
        return 0, 0, 0, pd.Series(), 0 # 승률 0 반환 추가

    final_return = cum_returns.iloc[-1] - 1

    # MDD 계산
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    mdd = drawdown.min()

    # 매매 횟수 (진입+청산 / 2)
    trades = position_series.diff().abs().sum() / 2

    # [추가] 승률(Win Rate) 계산
    # 보유 구간(1)들을 그룹화하여 각 매매별 수익률 계산
    if trades > 0:
        # 포지션이 변하는 지점을 기준으로 그룹 ID 생성
        trade_ids = (position_series != position_series.shift()).cumsum()
        # 포지션이 1인(보유 중인) 구간만 필터링
        holding_mask = position_series == 1
        if holding_mask.sum() > 0:
            # 각 매매(그룹)별로 누적 수익률 계산: (1+r1)*(1+r2)... - 1
            per_trade_returns = strategy_returns[holding_mask].groupby(trade_ids[holding_mask]).apply(lambda x: (1 + x).prod() - 1)
            # 0보다 큰 수익을 낸 횟수 / 전체 매매 횟수
            win_count = (per_trade_returns > 0).sum()
            win_rate = win_count / len(per_trade_returns)
        else:
            win_rate = 0
    else:
        win_rate = 0

    return final_return, mdd, trades, cum_returns, win_rate

# ==========================================
# 2. 데이터 준비 및 기간 분리
# ==========================================
TICKER = 'KS200' # (삼성전자:'005930', KOSPI200:'KS200', SK하이닉스:'000660')
START_DATE = '2015-01-01'
END_DATE = '2025-12-31'

print(f"[{TICKER}] 데이터 로딩 중...")
df_all = fdr.DataReader(TICKER, START_DATE, END_DATE)

# 등락률 계산 (Close 기준)
if 'Change' not in df_all.columns:
    df_all['Change'] = df_all['Close'].pct_change()

# 기간 분리
train_df = df_all.loc['2015-01-01':'2019-12-31'].copy()
test_df = df_all.loc['2020-01-01':'2025-12-31'].copy()

print(f"Train 기간: {train_df.index[0].date()} ~ {train_df.index[-1].date()}")
print(f"Test  기간: {test_df.index[0].date()} ~ {test_df.index[-1].date()}")

# ==========================================
# 3. [Train] 파라미터 최적화 (Grid Search)
# ==========================================
print(f"\n🚀 [Step 1] 훈련 데이터(Train) 최적화 시작...")

FAST_RANGE = range(5, 25, 5)
SLOW_RANGE = range(20, 60, 10)
SIGNAL_RANGE = range(5, 15, 2)
ALPHA_RANGE = [0.5, 0.7, 0.9]

best_train_result = {'score': -np.inf, 'params': None}

total_iter = len(ALPHA_RANGE) * len(FAST_RANGE) * len(SLOW_RANGE) * len(SIGNAL_RANGE)
pbar = tqdm(total=total_iter)

train_close = train_df['Close']
train_change = train_df['Change']

for alpha in ALPHA_RANGE:
    for fast in FAST_RANGE:
        ma_fast = calc_custom_ma(train_close, fast, alpha)

        for slow in SLOW_RANGE:
            if fast >= slow:
                pbar.update(len(SIGNAL_RANGE))
                continue

            ma_slow = calc_custom_ma(train_close, slow, alpha)
            macd_full = ma_fast - ma_slow

            for signal in SIGNAL_RANGE:
                signal_full = calc_custom_ma(macd_full, signal, alpha)

                # [변경] 반환값 unpacking에 win_rate(_) 추가
                ret, mdd, trades, _, _ = calculate_performance(train_change, macd_full, signal_full)

                if ret > best_train_result['score']:
                    best_train_result['score'] = ret
                    best_train_result['params'] = (alpha, fast, slow, signal)

                pbar.update(1)

pbar.close()

b_alpha, b_fast, b_slow, b_sig = best_train_result['params']
print(f"\n✅ 최적 파라미터 발견!")
print(f"   Alpha: {b_alpha}")
print(f"   Fast: {b_fast} / Slow: {b_slow} / Signal: {b_sig}")
print(f"   Train 수익률: {best_train_result['score']*100:.2f}%")


# ==========================================
# 4. [Test] 실전 백테스트 (Out-of-Sample)
# ==========================================
print("\n" + "="*60)
print("🕵️‍♀️ [Step 2] 검증 데이터(Test) 실전 테스트")
print("="*60)

# 1. 벤치마크 (Buy & Hold)
test_buy_hold = (1 + test_df['Change'].fillna(0)).cumprod()
bh_return = test_buy_hold.iloc[-1] - 1
bh_mdd = (test_buy_hold / test_buy_hold.cummax() - 1).min()

# 2. 표준 MACD (12, 26, 9)
std_ema12 = df_all['Close'].ewm(span=12, adjust=False).mean()
std_ema26 = df_all['Close'].ewm(span=26, adjust=False).mean()
std_macd = std_ema12 - std_ema26
std_sig = std_macd.ewm(span=9, adjust=False).mean()

# [변경] 표준 MACD의 모든 지표(MDD, Trades, WinRate) 수신
std_ret, std_mdd, std_trades, std_curve, std_win_rate = calculate_performance(
    df_all.loc[test_df.index, 'Change'],
    std_macd.loc[test_df.index],
    std_sig.loc[test_df.index]
)

# 3. 최적화 전략
final_ma_fast = calc_custom_ma(df_all['Close'], b_fast, b_alpha)
final_ma_slow = calc_custom_ma(df_all['Close'], b_slow, b_alpha)
final_macd = final_ma_fast - final_ma_slow
final_signal = calc_custom_ma(final_macd, b_sig, b_alpha)

# [변경] 최적화 모델의 모든 지표 수신
oos_ret, oos_mdd, oos_trades, oos_curve, oos_win_rate = calculate_performance(
    df_all.loc[test_df.index, 'Change'],
    final_macd.loc[test_df.index],
    final_signal.loc[test_df.index]
)

# ==========================================
# 5. 결과 출력 및 시각화
# ==========================================
print(f"📊 최종 성적표 (2020.01 ~ 2025.12)")
print(f"{'구분':<15} | {'수익률':<10} | {'MDD':<10} | {'매매횟수'}")
print("-" * 75)
print(f"{'표준 MACD':<15} | {std_ret*100:6.2f}%   | {std_mdd*100:6.2f}%   | {int(std_trades)}회")
print(f"{'최적화 모델':<15}| {oos_ret*100:6.2f}%  | {oos_mdd*100:6.2f}%   | {int(oos_trades)}회")
print("-" * 75)

# 그래프 그리기
plt.figure(figsize=(14, 6))

# 왼쪽: Train 결과
plt.subplot(1, 2, 1)
t_macd = final_macd.loc[train_df.index]
t_sig = final_signal.loc[train_df.index]
# unpacking 수정 (train graph용)
_, _, _, t_curve, _ = calculate_performance(train_change, t_macd, t_sig)
t_bh = (1 + train_change.fillna(0)).cumprod()

plt.plot(t_curve.index, t_curve, label='Optimized', color='red')
plt.plot(t_bh.index, t_bh, label='Buy&Hold', color='gray', linestyle='--')
plt.title(f'[TRAIN] 2015-2019 (Fitting)\nReturn: {(t_curve.iloc[-1]-1)*100:.1f}%')
plt.legend()
plt.grid(True, alpha=0.3)

# 오른쪽: Test 결과
plt.subplot(1, 2, 2)

plt.plot(oos_curve.index, oos_curve, label='Optimized (OOS)', color='blue')
plt.plot(test_buy_hold.index, test_buy_hold, label='Buy&Hold', color='gray', linestyle='--')
plt.title(f'[TEST] 2020-2025 (Validation)\nReturn: {oos_ret*100:.1f}%')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 6. 추가: 파라미터 민감도 히트맵 시각화
# ==========================================
print("\n🔥 [Step 3] 파라미터 민감도 히트맵 생성 중...")

heatmap_data = []
for fast in FAST_RANGE:
    row = []
    ma_fast = calc_custom_ma(train_close, fast, b_alpha)
    for slow in SLOW_RANGE:
        if fast >= slow:
            row.append(np.nan)
            continue

        ma_slow = calc_custom_ma(train_close, slow, b_alpha)
        macd_full = ma_fast - ma_slow
        signal_full = calc_custom_ma(macd_full, b_sig, b_alpha)

        # unpacking 수정
        ret, _, _, _, _ = calculate_performance(train_change, macd_full, signal_full)
        row.append(ret)
    heatmap_data.append(row)

df_heatmap = pd.DataFrame(heatmap_data, index=FAST_RANGE, columns=SLOW_RANGE)

plt.figure(figsize=(10, 8))
sns.heatmap(df_heatmap, annot=True, fmt=".1%", cmap='RdYlGn', center=0)
plt.title(f"Parameter Sensitivity Analysis (Alpha:{b_alpha}, Signal:{b_sig})\n[Train Period Yield]")
plt.xlabel("Slow Period")
plt.ylabel("Fast Period")
plt.show()