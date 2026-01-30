import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from strategy_core import calc_ma_recursive, calculate_performance

# ==========================================
# 0. 사용자 설정
# ==========================================
TICKER = 'ks200'
T_NAME = "KOSPI 200"
TEST_START = "2020-01-01"
TEST_END   = "2025-12-31"

# 최적화 결과값 (히트맵이나 Train에서 찾은 값)
B_FAST, B_SLOW, B_SIG, B_ALPHA = 20, 40, 5, 0.35 

def plot_cum_return_compare_test(test_buy_hold, std_curve, oos_curve, std_ret, oos_ret, 
                                 b_fast, b_slow, b_sig, b_alpha, 
                                 title_prefix="[TEST]", savepath=None, show=True):
    
    # 1. 인덱스 정렬 (FFILL로 빈틈 없이 메꿈)
    bh = test_buy_hold.ffill()
    st = std_curve.ffill()
    op = oos_curve.ffill()
    bh_return = float(bh.iloc[-1] - 1)

    # 2. 그래프 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(bh.index, bh, label="Buy&Hold (Market)", color='gray', linestyle="--", alpha=0.7)
    plt.plot(st.index, st, label="Standard MACD (12,26,9)", color='blue', alpha=0.8)
    plt.plot(op.index, op, label=f"Optimized MACD (F={b_fast}, S={b_slow}, Sig={b_sig}, α={b_alpha})", color='red', linewidth=2)

    plt.title(
        f"{title_prefix} Cumulative Return Comparison\n"
        f"Std={std_ret*100:.1f}% vs Opt={oos_ret*100:.1f}% (BH={bh_return*100:.1f}%)",
        fontsize=14, fontweight='bold'
    )
    plt.ylabel("Cumulative Growth (Start=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    if savepath:
        base, _ = os.path.splitext(savepath)
        plt.savefig(f"{base}.pdf", bbox_inches="tight")
        plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
        print(f"💾 Saved: {base}.pdf / {base}.png")

    if show:
        plt.show()
    plt.close()

# ==========================================
# 실행 부분: 이중 shift 문제를 해결한 정밀 계산
# ==========================================
if __name__ == "__main__":
    print(f"🚀 [{TICKER}] 수익률 정밀 분석 시작...")
    
    # 1. 데이터 로드 (시작점을 2015년으로 설정하여 MA 초기값 일치)
    df_all = fdr.DataReader(TICKER, "2015-01-01", TEST_END)
    
    # 2. 지표 전구간 계산
    # 표준
    std_m = df_all['Close'].ewm(span=12, adjust=False).mean() - df_all['Close'].ewm(span=26, adjust=False).mean()
    std_s = std_m.ewm(span=9, adjust=False).mean()
    # 최적
    opt_m = calc_ma_recursive(df_all['Close'], B_FAST, B_ALPHA) - calc_ma_recursive(df_all['Close'], B_SLOW, B_ALPHA)
    opt_s = calc_ma_recursive(opt_m, B_SIG, B_ALPHA)

    # 3. [중요] 테스트 구간 데이터만 추출
    # calculate_performance 내부에서 shift(1)을 하므로, 여기서 미리 shift 하면 안 됨!
    test_df = df_all.loc[TEST_START:TEST_END].copy()
    if 'Change' not in test_df.columns:
        test_df['Change'] = test_df['Close'].pct_change()

    # 4. 성능 계산 (strategy_core의 함수 사용)
    # calculate_performance(수익률, MACD, Signal) 순서
    std_ret_val, std_curve, _ = calculate_performance(
        test_df['Change'], 
        std_m.loc[test_df.index], 
        std_s.loc[test_df.index]
    )
    
    oos_ret_val, oos_curve, _ = calculate_performance(
        test_df['Change'], 
        opt_m.loc[test_df.index], 
        opt_s.loc[test_df.index]
    )
    
    # 시장 수익률 (Buy & Hold)
    test_bh = (1 + test_df['Change'].fillna(0)).cumprod()

    # 5. 시각화 호출
    plot_cum_return_compare_test(
        test_buy_hold=test_bh,
        std_curve=std_curve,
        oos_curve=oos_curve,
        std_ret=std_ret_val,
        oos_ret=oos_ret_val,
        b_fast=B_FAST, b_slow=B_SLOW, b_sig=B_SIG, b_alpha=B_ALPHA,
        title_prefix=f"[{T_NAME}] {TEST_START}~{TEST_END}",
        savepath="KOSPI_cumulative_return_compare_TEST",
        show=True
    )