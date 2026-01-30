import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from strategy_core import calc_ma_recursive, calculate_performance 

# [1. 분석 대상 리스트] - 이미지와 동일하게 구성
analysis_targets = [
    {
        'symbol': 'KS200', 
        'name': 'KOSPI 200',
        'alpha': 0.35,
        'fast_list': [5, 10, 15, 20],
        'slow_list': [20, 30, 40, 50, 60]
    },
    {
        'symbol': '000660', 
        'name': 'SK HYNIX',
        'alpha': 0.8,
        # 목표 이미지처럼 Fast가 25까지 있고 Slow보다 큰 경우도 포함
        'fast_list': [5, 10, 15, 20, 25], 
        'slow_list': [20, 30, 40, 50, 60]
    }
]

SIGNAL_LIST = [5, 7, 9, 11, 13]
TRAIN_START, TRAIN_END = "2015-01-01", "2019-12-31"

def plot_heatmap_fast_slow(df_all, start, end, alpha_fixed, fast_list, slow_list, signal_list, title="", savepath=None):
    
    # 1. 57.7% / 165.9%를 위해 전체 데이터에서 Change를 먼저 계산
    df_all_calc = df_all.copy()
    df_all_calc["Change"] = df_all_calc["Close"].pct_change()
    
    # 분석 타겟 구간의 인덱스 추출
    target_idx = df_all_calc.loc[start:end].index
    change_series = df_all_calc.loc[target_idx, "Change"].fillna(0.0)
    
    # 데이터프레임 구조 (Fast x Slow)
    mat = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)
    best_sig = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)
    
    print(f"🔍 {title} 연산 중...")
    for fast in fast_list:
        # MA는 반드시 df_all(전체) 기준으로 계산해야 초기값이 정확함
        ma_fast = calc_ma_recursive(df_all["Close"], fast, alpha_fixed)
        
        for slow in slow_list:
            # 🔥 [수정] 목표 이미지(2번째)처럼 Fast > Slow 인 경우도 계산하도록 제약 제거
            ma_slow = calc_ma_recursive(df_all["Close"], slow, alpha_fixed)
            macd_all = ma_fast - ma_slow
            
            best_val = -np.inf
            best_s = np.nan

            for sigN in signal_list:
                sig_all = calc_ma_recursive(macd_all, sigN, alpha_fixed)
                
                # 인덱스를 target_idx로 강제 슬라이싱하여 calculate_performance에 전달
                ret, _, _ = calculate_performance(
                    change_series, 
                    macd_all.loc[target_idx], 
                    sig_all.loc[target_idx]
                )
                
                if ret is not None and ret > best_val:
                    best_val = ret
                    best_s = sigN
            
            mat.loc[fast, slow] = best_val
            best_sig.loc[fast, slow] = best_s

    # 2. 시각화 (이미지 2와 100% 일치 레이아웃)
    plt.figure(figsize=(11, 7))
    
    # mat.values를 그대로 시각화 (Y축 index가 아래에서 위로 증가하는 구조)
    im = plt.imshow(mat.values.astype(float), origin="lower", aspect="auto", cmap="coolwarm")
    
    plt.xticks(range(len(slow_list)), slow_list)
    plt.yticks(range(len(fast_list)), fast_list)
    plt.xlabel("Slow")
    plt.ylabel("Fast")
    plt.title(title, fontsize=12, fontweight='bold')
    
    # 셀 값 표시 (이미지 2의 숫자들 51.1%, 119.7% 등 재현)
    for i, f_val in enumerate(fast_list):
        for j, s_val in enumerate(slow_list):
            val = mat.loc[f_val, s_val]
            if np.isfinite(val):
                plt.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=9, color="black")

    # 최고 수익률 지점 강조 (sig=11 등 표시)
    if np.isfinite(mat.values.astype(float)).any():
        valid_mat = mat.values.astype(float)
        r_idx, c_idx = np.unravel_index(np.nanargmax(valid_mat), mat.shape)
        plt.scatter([c_idx], [r_idx], s=150, facecolors='none', edgecolors='black', linewidths=2)
        plt.text(c_idx, r_idx, f"\n★\n(sig={int(best_sig.iloc[r_idx, c_idx])})", 
                 ha="center", va="center", color="white", fontweight="bold")

    plt.colorbar(im).set_label("Return")
    plt.tight_layout()
    if savepath: plt.savefig(f"{savepath}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    for target in analysis_targets:
        # 2015-01-01부터 데이터를 로드
        df_all = fdr.DataReader(target['symbol'], "2015-01-01", TRAIN_END)
        plot_heatmap_fast_slow(
            df_all=df_all,
            start=TRAIN_START, end=TRAIN_END,
            alpha_fixed=target['alpha'],
            fast_list=target['fast_list'],
            slow_list=target['slow_list'],
            signal_list=SIGNAL_LIST,
            title=f"[{target['name']}][TRAIN] Heatmap (alpha={target['alpha']})",
            savepath=f"Heatmap_{target['name']}"
        )