import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from strategy_core import calc_ma_recursive, calculate_performance # 엔진 로드

# ==========================================
# 0. 사용자 설정 (이 파일만 개별 실행 시 적용)
# ==========================================
TICKER = 'ks200'
ALPHA_FIXED = 0.35  # 최적화에서 찾은 Alpha 값
FAST_LIST   = [5, 10, 15, 20]
SLOW_LIST   = [20, 30, 40, 50, 60]
SIGNAL_LIST = [5, 7, 9, 11, 13]
TRAIN_START, TRAIN_END = "2015-01-01", "2019-12-31"

def plot_heatmap_fast_slow(df_all, start, end, alpha_fixed, fast_list, slow_list, signal_list, title="", savepath=None):
    # 1. 기간 데이터 추출 및 준비
    df_period = df_all.loc[start:end].copy()
    if len(df_period) == 0: raise ValueError("해당 기간에 데이터가 없습니다.")
    
    change = df_period["Change"].fillna(0.0)
    mat = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)
    best_sig = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)
    
    # 속도 향상을 위한 MA 캐싱
    ma_cache = {}

    def get_ma(series, n, alpha):
        key = (n, alpha)
        if key not in ma_cache:
            ma_cache[key] = calc_ma_recursive(series, n, alpha)
        return ma_cache[key]

    # 2. 루프 연산 (Fast x Slow x Signal)
    print("🔍 히트맵 연산 중... (각 셀별 최적 Signal 탐색)")
    for fast in fast_list:
        ma_fast = get_ma(df_all["Close"], fast, alpha_fixed)
        for slow in slow_list:
            if slow <= fast:
                continue

            ma_slow = get_ma(df_all["Close"], slow, alpha_fixed)
            macd = ma_fast - ma_slow
            
            best_val = -np.inf
            best_s = np.nan

            for sigN in signal_list:
                sig_series = calc_ma_recursive(macd, sigN, alpha_fixed)
                # 성능 평가 (Return만 사용)
                ret, _, _ = calculate_performance(
                    change, 
                    macd.loc[df_period.index], 
                    sig_series.loc[df_period.index]
                )
                
                if ret > best_val:
                    best_val = ret
                    best_s = sigN
            
            mat.loc[fast, slow] = best_val
            best_sig.loc[fast, slow] = best_s

    # 3. 시각화 (올려주신 디자인 폼 유지)
    plt.figure(figsize=(11, 7))
    im = plt.imshow(mat.values.astype(float), origin="lower", aspect="auto", cmap="coolwarm")
    
    plt.xticks(range(len(slow_list)), slow_list)
    plt.yticks(range(len(fast_list)), fast_list)
    plt.xlabel("Slow Period")
    plt.ylabel("Fast Period")
    plt.title(title)
    
    # 수치 표시 (Annotation)
    for i in range(len(fast_list)):
        for j in range(len(slow_list)):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=9)

    # 베스트 셀 표시 (별표 마크)
    if np.isfinite(mat.values.astype(float)).any():
        valid_mat = mat.values.astype(float)
        r, c = np.unravel_index(np.nanargmax(valid_mat), mat.shape)
        plt.scatter([c], [r], s=150, facecolors='none', edgecolors='black', linewidths=2)
        plt.text(c, r, f"\n★\n(sig={int(best_sig.iloc[r,c])})", 
                 ha="center", va="center", color="white", fontweight="bold")

    plt.colorbar(im).set_label("Return")
    plt.tight_layout()

    # 4. 저장 및 출력
    if savepath:
        plt.savefig(f"{savepath}.pdf", bbox_inches="tight")
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
        print(f"💾 Saved: {savepath}.pdf / .png")

    plt.show()

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    print(f"🚀 [{TICKER}] 히트맵 분석 시작...")
    df_all = fdr.DataReader(TICKER, "2015-01-01")
    df_all["Change"] = df_all["Close"].pct_change()
    
    plot_heatmap_fast_slow(
        df_all=df_all,
        start=TRAIN_START,
        end=TRAIN_END,
        alpha_fixed=ALPHA_FIXED,
        fast_list=FAST_LIST,
        slow_list=SLOW_LIST,
        signal_list=SIGNAL_LIST,
        title=f"MACD Optimization Heatmap (Alpha={ALPHA_FIXED})",
        savepath="KOSPI_HEATMAP_optimized"
    )