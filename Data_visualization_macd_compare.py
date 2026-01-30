import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from strategy_core import calc_ma_recursive # 연산 함수 불러오기

# ==========================================
# 0. 설정 (종목 및 파라미터)
# ==========================================
TICKER = 'ks200'
T_NAME = "KOSPI 200"
PLOT_START, PLOT_END = "2020-02-01", "2025-12-31"

# 최적화에서 찾았던 파라미터 입력
B_FAST, B_SLOW, B_SIG, B_ALPHA = 20, 40, 5, 0.35 

# -----------------------------
# 시각화 함수: 올려주신 mplfinance 로직 그대로
# -----------------------------
def plot_candle_macd_dashboard(df_ohlcv, macd, signal, title, savepath=None):
    dfp = df_ohlcv.copy()
    macd = macd.reindex(dfp.index)
    signal = signal.reindex(dfp.index)
    hist = macd - signal

    # 신호 계산
    pos = (macd > signal).astype(int)
    chg = pos.diff().fillna(0)
    buy_price = dfp["Low"].where(chg == 1) * 0.995
    sell_price = dfp["High"].where(chg == -1) * 1.005

    apds = [
        mpf.make_addplot(buy_price, type='scatter', markersize=70, marker='^', panel=0, color='green'),
        mpf.make_addplot(sell_price, type='scatter', markersize=70, marker='v', panel=0, color='red'),
        mpf.make_addplot(macd, panel=2),
        mpf.make_addplot(signal, panel=2),
        mpf.make_addplot(hist, panel=2, type='bar', alpha=0.4),
    ]

    fig, _ = mpf.plot(
        dfp, type='candle', volume=True, addplot=apds,
        panel_ratios=(6, 2, 3), title=title, style='yahoo',
        figsize=(14, 9), returnfig=True
    )

    if savepath:
        fig.tight_layout()
        fig.savefig(f"{savepath}.pdf", bbox_inches="tight")
        fig.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
        print(f"💾 Saved: {savepath}.pdf / .png")

    plt.show()

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 로딩
    df_all = fdr.DataReader(TICKER, "2015-01-01", PLOT_END)
    df_plot = df_all.loc[PLOT_START:PLOT_END].copy()

    # 2. 표준 MACD (12, 26, 9) - Pandas ewm 사용
    std_macd = df_all['Close'].ewm(span=12, adjust=False).mean() - df_all['Close'].ewm(span=26, adjust=False).mean()
    std_sig = std_macd.ewm(span=9, adjust=False).mean()

    # 3. 최적화 MACD - 엔진 함수 사용
    opt_macd = calc_ma_recursive(df_all['Close'], B_FAST, B_ALPHA) - calc_ma_recursive(df_all['Close'], B_SLOW, B_ALPHA)
    opt_sig = calc_ma_recursive(opt_macd, B_SIG, B_ALPHA)

    # 차트 1: 표준 MACD
    plot_candle_macd_dashboard(
        df_plot, std_macd.loc[df_plot.index], std_sig.loc[df_plot.index],
        title=f"[{T_NAME}] Standard MACD (12,26,9)",
        savepath="KOSPI_MACD_standard"
    )

    # 차트 2: 최적화 MACD
    plot_candle_macd_dashboard(
        df_plot, opt_macd.loc[df_plot.index], opt_sig.loc[df_plot.index],
        title=f"[{T_NAME}] Optimized MACD (F={B_FAST}, S={B_SLOW}, Sig={B_SIG}, α={B_ALPHA})",
        savepath="KOSPI_MACD_optimized"
    )