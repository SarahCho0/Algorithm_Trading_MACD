import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from strategy_core import calc_ma_recursive

# [1. 분석 대상 리스트 설정]
analysis_targets = [
    {
        'symbol': 'KS200', 
        'name': 'KOSPI 200',
        'params': {'fast': 20, 'slow': 40, 'signal': 5, 'alpha': 0.35}
    },
    {
        'symbol': '000660', 
        'name': 'SK HYNIX',
        'params': {'fast': 20, 'slow': 50, 'signal': 11, 'alpha': 0.8}
    }
]

# [2. 차트 생성 함수 - 기존 로직 유지]
def plot_candle_macd_dashboard(df_ohlcv, macd, signal, title, savepath=None):
    dfp = df_ohlcv.copy()
    macd = macd.reindex(dfp.index)
    signal = signal.reindex(dfp.index)
    hist = macd - signal

    # 신호 계산 (어제 신호로 오늘 화살표 표시)
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
        print(f"💾 저장 완료: {savepath}.pdf / .png")

    plt.show()
    plt.close(fig)

# [3. 실행부: 2개 종목 순회]
if __name__ == "__main__":
    PLOT_START = "2020-02-01"
    PLOT_END   = "2025-12-31"

    for target in analysis_targets:
        S = target['symbol']
        N = target['name']
        P = target['params']
        
        print(f"\n🚀 {N} ({S}) 분석 시작...")
        
        # 데이터 로드 (계산용으로 조금 앞서서 가져옴)
        df_all = fdr.DataReader(S, "2019-01-01", PLOT_END)
        df_plot = df_all.loc[PLOT_START:PLOT_END].copy()

        # --- (A) 표준 MACD (12, 26, 9) 계산 ---
        std_m = df_all['Close'].ewm(span=12, adjust=False).mean() - df_all['Close'].ewm(span=26, adjust=False).mean()
        std_s = std_m.ewm(span=9, adjust=False).mean()

        print(f"  > {N} 표준 차트 그리는 중...")
        plot_candle_macd_dashboard(
            df_plot, 
            std_m.loc[df_plot.index], 
            std_s.loc[df_plot.index],
            title=f"[{N}] Standard MACD (12,26,9) | {PLOT_START}~{PLOT_END}",
            savepath=f"MACD_Standard_{N.replace(' ', '_')}"
        )

        # --- (B) 최적화 MACD 계산 ---
        opt_m = calc_ma_recursive(df_all['Close'], P['fast'], P['alpha']) - \
                calc_ma_recursive(df_all['Close'], P['slow'], P['alpha'])
        opt_s = calc_ma_recursive(opt_m, P['signal'], P['alpha'])

        print(f"  > {N} 최적화 차트 그리는 중...")
        plot_candle_macd_dashboard(
            df_plot, 
            opt_m.loc[df_plot.index], 
            opt_s.loc[df_plot.index],
            title=f"[{N}] Optimized MACD (F={P['fast']}, S={P['slow']}, Sig={P['signal']}, α={P['alpha']})",
            savepath=f"MACD_Optimized_{N.replace(' ', '_')}"
        )

    print("\n✅ 모든 종목의 분석 대시보드가 생성되었습니다.")