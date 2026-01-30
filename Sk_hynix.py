import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ==========================================
# 1. 핵심 함수 정의
# ==========================================


def calc_custom_ma_recursive(series, n, alpha):
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
TICKER = '000660' # (삼성전자:'005930', KOSPI200:'KS200', SK하이닉스:'000660')
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
ALPHA_RANGE = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

best_train_result = {'score': -np.inf, 'params': None}
total_iter = len(ALPHA_RANGE) * len(FAST_RANGE) * len(SLOW_RANGE) * len(SIGNAL_RANGE)
pbar = tqdm(total=total_iter)

train_close = train_df['Close']
train_change = train_df['Change']

for alpha in ALPHA_RANGE:
    for fast in FAST_RANGE:
        ma_fast = calc_custom_ma_recursive(train_close, fast, alpha)
        for slow in SLOW_RANGE:
            if fast >= slow:
                pbar.update(len(SIGNAL_RANGE))
                continue
            ma_slow = calc_custom_ma_recursive(train_close, slow, alpha)
            macd_full = ma_fast - ma_slow
            for signal in SIGNAL_RANGE:
                signal_full = calc_custom_ma_recursive(macd_full, signal, alpha)
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
    print(f"\n최적 파라미터 발견!")
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
    final_ma_fast = calc_custom_ma_recursive(df_all['Close'], b_fast, b_alpha)
    final_ma_slow = calc_custom_ma_recursive(df_all['Close'], b_slow, b_alpha)
    final_macd = final_ma_fast - final_ma_slow
    final_signal = calc_custom_ma_recursive(final_macd, b_sig, b_alpha)

    oos_ret, oos_curve, _ = calculate_performance(
        df_all.loc[test_df.index, 'Change'],
        final_macd.loc[test_df.index],
        final_signal.loc[test_df.index]
    )

    # ==========================================
    # 5. 결과 출력 및 시각화
    # ==========================================
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

# ==========================================
# 6. 추가: 파라미터 민감도 히트맵 시각화
# ==========================================
print("\n🔥 [Step 2] 파라미터 민감도 히트맵 생성 중...")


def plot_heatmap_fast_slow(
    df_all: pd.DataFrame,
    start: str,
    end: str,
    alpha_fixed: float,
    fast_list,
    slow_list,
    signal_list,
    title: str = "",
    savepath: str | None = None,
    ma_func=None,              # ✅ 커스텀 MA 함수 주입 가능
    max_signal_offset: int = 1,# ✅ signal <= fast - max_signal_offset
    show_annot: bool = True,   # ✅ 셀 값 표시 여부
):
    """
    fast x slow 그리드에서, 각 셀마다 signal_list를 훑어 '최고 수익률'을 만드는 signal을 선택하고
    그 최고 수익률을 heatmap으로 그립니다.

    - df_all: Close, Change 컬럼 필요
    - start/end: 표시할 기간 (OOS/특정 구간)
    - alpha_fixed: alpha 고정
    - ma_func: 이동평균 함수. 기본은 calc_custom_ma_recursive 사용 권장
    """

    if ma_func is None:
        # 기본값: 사용자가 주는 calc_custom_ma_recursive를 쓸 수 있게 이름만 맞춰둠
        ma_func = calc_custom_ma_recursive

    # --- 입력 검증 ---
    required_cols = {"Close", "Change"}
    missing = required_cols - set(df_all.columns)
    if missing:
        raise ValueError(f"df_all에 필요한 컬럼이 없습니다: {missing}")

    df_period = df_all.loc[start:end].copy()
    if len(df_period) == 0:
        raise ValueError("start~end 기간에 해당하는 데이터가 없습니다.")

    change = df_period["Change"].fillna(0.0)

    fast_list = list(fast_list)
    slow_list = list(slow_list)
    signal_list = list(signal_list)

    mat = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)
    best_sig = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)

    close_all = df_all["Close"]

    # --- 속도 개선: alpha 고정이므로 period별 MA를 캐싱 ---
    ma_cache = {}  # key: n -> Series

    def get_ma(series: pd.Series, n: int, alpha: float) -> pd.Series:
        # close_all에 대해서만 캐싱(가장 많이 호출됨)
        key = ("close", n, alpha)
        if series is close_all:
            if key not in ma_cache:
                ma_cache[key] = ma_func(series, n, alpha)
            return ma_cache[key]
        # macd_series는 n, alpha별로 캐싱하기 애매하니 기본 계산(필요하면 추가 캐싱 가능)
        return ma_func(series, n, alpha)

    # --- 계산 ---
    for fast in fast_list:
        ma_fast = get_ma(close_all, fast, alpha_fixed)

        for slow in slow_list:
            # ✅ 논리 제약: fast/slow 간격 + fast<slow
            if slow < fast:
                mat.loc[fast, slow] = np.nan
                best_sig.loc[fast, slow] = np.nan
                continue

            ma_slow = get_ma(close_all, slow, alpha_fixed)
            macd = ma_fast - ma_slow

            best_val = -np.inf
            best_s = np.nan

            for sigN in signal_list:
                sig = ma_func(macd, sigN, alpha_fixed)

                # (주의) calculate_performance는 (ret, cum, win_rate) 또는 (ret, cum, win, trades) 등
                # 사용자 구현에 따라 반환이 다를 수 있으니 ret만 안전하게 받기
                out = calculate_performance(
                    change_series=change,
                    macd_series=macd.loc[df_period.index],
                    signal_series=sig.loc[df_period.index]
                )
                ret = out[0]

                if ret is None or np.isnan(ret) or np.isinf(ret):
                    continue

                if ret > best_val:
                    best_val = ret
                    best_s = sigN

            # signal 제약 때문에 유효한 sig가 하나도 없을 수 있음
            if best_val == -np.inf:
                mat.loc[fast, slow] = np.nan
                best_sig.loc[fast, slow] = np.nan
            else:
                mat.loc[fast, slow] = best_val
                best_sig.loc[fast, slow] = best_s

    # --- Plot ---
    plt.figure(figsize=(11, 7))
    im = plt.imshow(
        mat.values,
        origin="lower",
        aspect="auto",
        cmap="coolwarm"
    )

    plt.xticks(range(len(slow_list)), slow_list)
    plt.yticks(range(len(fast_list)), fast_list)
    plt.xlabel("Slow")
    plt.ylabel("Fast")
    plt.title(title)

    plt.grid(False)
    plt.minorticks_off()

    cbar = plt.colorbar(im)
    cbar.set_label("Return")

    # ✅ 셀 annotation
    if show_annot:
        for i in range(len(fast_list)):
            for j in range(len(slow_list)):
                val = mat.iloc[i, j]
                if np.isfinite(val):
                    plt.text(
                        j, i, f"{val*100:.1f}%",
                        ha="center", va="center",
                        fontsize=9, color="black"
                    )

    # ✅ best cell 표시 (NaN만 있으면 스킵)
    if np.isfinite(mat.values).any():
        r, c = np.unravel_index(np.nanargmax(mat.values), mat.shape)
        plt.scatter([c], [r], s=120, facecolors='none', edgecolors='black', linewidths=2)

        best_sig_val = best_sig.iloc[r, c]
        if np.isfinite(best_sig_val):
            plt.text(
                c, r, f"\n★\n(sig={int(best_sig_val)})",
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold"
            )

        print(f"[BEST] fast={fast_list[r]}, slow={slow_list[c]}, sig={best_sig_val}, alpha={alpha_fixed}")
        print(f"       Return = {mat.iloc[r, c]*100:.2f}%")
    else:
        print("[WARN] 모든 셀이 NaN입니다. (제약 조건이 너무 빡세거나 기간/범위가 문제일 수 있음)")

    plt.tight_layout()

    # ✅ 저장 로직 (show 전에!)
    if savepath:
        base = savepath
        if base.lower().endswith(".pdf"):
            base = base[:-4]
        plt.savefig(f"{base}.pdf", bbox_inches="tight")           # PDF (벡터)
        plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight")  # PNG (고해상도)
        print(f"💾 Saved: {base}.pdf / {base}.png")

    plt.show()
    plt.close()

    return mat, best_sig

FAST_LIST   = [5,10,15,20,25]
SLOW_LIST   = [20,30,40,50,60]
SIGNAL_LIST = [5,7,9,11,13,15]

mat_ret, best_sig = plot_heatmap_fast_slow(
    df_all=df_all,
    start="2015-01-01",
    end="2019-12-31", # ★ 반드시 TRAIN
    alpha_fixed=b_alpha,
    fast_list=FAST_LIST,
    slow_list=SLOW_LIST,
    signal_list=SIGNAL_LIST,
    title=f"[KOSPI200][TRAIN] Heatmap (alpha={b_alpha})",
    # savepath=f"KOSPI_HEATMAP_origin"
)


# ==========================================
# 7. 추가: 누적수익률 비교 그래프
# ==========================================
print("\n🔥 [Step 3] 누적수익률 비교 그래프 생성 중...")
def plot_cum_return_compare_test(
    test_buy_hold,
    std_curve,
    oos_curve,
    std_ret,
    oos_ret,
    b_fast, b_slow, b_sig, b_alpha,
    title_prefix="[TEST] 2020-2025",
    savepath=None,
    show=True
):
    import os, numpy as np
    import matplotlib.pyplot as plt

    common_idx = test_buy_hold.index.intersection(std_curve.index).intersection(oos_curve.index)
    if len(common_idx) == 0:
        raise ValueError("세 곡선의 공통 인덱스가 없습니다.")

    bh = test_buy_hold.loc[common_idx].replace([np.inf, -np.inf], np.nan).ffill()
    st = std_curve.loc[common_idx].replace([np.inf, -np.inf], np.nan).ffill()
    op = oos_curve.loc[common_idx].replace([np.inf, -np.inf], np.nan).ffill()

    # ✅ 여기서 자동 계산
    bh_return = float(bh.iloc[-1] - 1)

    plt.figure(figsize=(14, 6))
    plt.plot(bh.index, bh, label="Buy&Hold", linestyle="--")
    plt.plot(st.index, st, label="Standard MACD (12,26,9)")
    plt.plot(op.index, op, label=f"Optimized MACD (fast={b_fast}, slow={b_slow}, sig={b_sig}, α={b_alpha})")

    plt.title(
        f"{title_prefix} Cumulative Return Comparison\n"
        f"Std={std_ret*100:.1f}% vs Opt={oos_ret*100:.1f}% (BH={bh_return*100:.1f}%)"
    )
    plt.ylabel("Cumulative Growth (Start=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    if savepath:
        base, ext = os.path.splitext(savepath)
        if ext == "":
            plt.savefig(f"{savepath}.pdf", bbox_inches="tight")
            plt.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
            print(f"💾 Saved: {savepath}.pdf / {savepath}.png")
        else:
            if ext.lower() == ".png":
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
            else:
                plt.savefig(savepath, bbox_inches="tight")
            print(f"💾 Saved: {savepath}")

    if show:
        plt.show()
    plt.close()

    return bh_return

plot_cum_return_compare_test(
    test_buy_hold=test_buy_hold,
    std_curve=std_curve,
    oos_curve=oos_curve,
    std_ret=std_ret,
    oos_ret=oos_ret,
    b_fast=b_fast, b_slow=b_slow, b_sig=b_sig, b_alpha=b_alpha,
    # savepath="KOSPI_cumulative_return_compare_TEST_2020_2025",  # 확장자 없이
    show=True
)
