# Algorithm Trading MACD Optimization

이 프로젝트는 **사용자 정의 가중 이동평균(Custom Weighted Moving Average)** 을 활용하여 MACD 지표를 최적화하고, KOSPI200 및 주요 종목(삼성전자, SK하이닉스)에 대한 백테스트를 수행하는 알고리즘 트레이딩 도구입니다.

## 📌 주요 기능
* **Grid Search 최적화**: 2015~2019년 데이터를 바탕으로 최적의 Alpha, Fast/Slow 기간, Signal 기간을 탐색합니다.
* **성능 평가 지표**: 누적 수익률, MDD(최대 낙폭), 승률(Win Rate), 매매 횟수를 산출합니다.
* **Out-of-Sample 검증**: 최적화된 파라미터를 2020~2025년 데이터에 적용하여 실전 성능을 검증합니다.
* **시각화**: 수익률 곡선 비교 차트 및 파라미터 민감도 분석 히트맵을 생성합니다.

## 🛠 설치 및 실행 방법

1.  **레포지토리 클론**
    ```bash
    git clone [https://github.com/SarahCho0/Algorithm_Trading_MACD.git](https://github.com/SarahCho0/Algorithm_Trading_MACD.git)
    cd Algorithm_Trading_MACD
    ```

2.  **필수 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

3.  **코드 실행**
    ```bash
    python Kospi200_custom.py
    # 또는 Samsung_custom.py / Sk_hynix.py / Data_visualization_custom.py / Data_visualization_MACD.py 실행
    ```

## 📊 분석 결과 예시
* **표준 MACD vs 최적화 모델**: 정량적 지표 비교 테이블 제공
* **수익률 차트**: 벤치마크(Buy & Hold) 대비 전략 성과 시각화
* **히트맵**: 파라미터 조합에 따른 수익률 변동성 확인
