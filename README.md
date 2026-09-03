# 신용 위험 모델링 파이프라인

10,000건의 재현 가능한 가상 금융 데이터를 사용해, 규칙 기반 분류와 로지스틱 회귀·랜덤 포레스트, 그리고 리지·라쏘 회귀를 비교하는 지도학습 포트폴리오 프로젝트입니다.

## 범위

분류 모델은 연체 위험의 **추가 심사 우선순위**를 탐색하는 예시입니다. 실제 고객을 자동 승인·거절하지 않으며, `predict_proba()`는 보정된 PD가 아닌 모델 추정 위험 점수로 해석합니다. 데이터와 결과는 교육용 가상 환경에 한정됩니다.

## 빠른 시작

Python 3.10 이상이 필요합니다.

```bash
uv sync
uv run python data_gen.py
uv run python train.py
uv run python -m pytest -v
```

`pip` 환경에서는 `pip install -r requirements.txt` 후 같은 명령을 `python`으로 실행할 수 있습니다. `finance_data.csv`와 예측 CSV는 생성 파일이며 Git에서 제외됩니다.

## 평가 대상

- 분류: 여섯 규칙 기반 기준선, 클래스 가중치를 적용한 로지스틱 회귀, 기본·튜닝 랜덤 포레스트
- 회귀: 리지·라쏘의 alpha 민감도와 holdout RMSE·MAE·R²
- 평가: 고정 80:20 학습·테스트 분할, 학습 데이터 내부 5-fold CV, Pipeline 내부 중앙값 대치와 스케일링

로지스틱 회귀는 최신 holdout에서 ROC-AUC와 F1이 가장 높았고, 튜닝 랜덤 포레스트는 비선형 모델의 비교 대상으로 유지합니다. 정확한 재생성 결과는 [artifacts/metrics.json](artifacts/metrics.json), 실험 설계와 해석은 [docs/model_selection.md](docs/model_selection.md)에서 확인합니다.

## 프로젝트 구조

```text
data_gen.py                    # 재현 가능한 가상 데이터 생성
train.py                       # 명령줄 실행 진입점
credit_risk/
  data.py                      # 스키마, 검증, 데이터 분할
  preprocessing.py             # 공통 누수 방지 전처리
  classification.py            # 학습, 교차검증, 임계값, 평가
  regression.py                # 규제 회귀 모델 선택
  reporting.py                 # JSON, CSV, 그래프 저장
  workflow.py                  # 실행 흐름 조립
docs/model_selection.md        # 방법론과 실험 선택 근거
artifacts/                     # 커밋된 지표 기록과 설명용 그래프
tests/                         # 동작 중심 회귀 테스트
```

## 재현성 및 산출물

`random_state=42`는 데이터 생성, 분할, 모델 난수를 고정합니다. `artifacts/metrics.json`은 기계가 읽을 수 있는 실행 기록이며, 커밋된 PNG는 프로젝트를 실행하지 않아도 현재 실험을 확인할 수 있게 합니다. 실행 시간은 하드웨어에 따라 달라지므로 문서에는 정확한 수치를 싣지 않고, 동일한 최대 2,000행 학습 특성 배치 조건과 함께 해당 artifact에만 기록합니다.

## 한계

- 가상 데이터로는 실제 신용 의사결정이나 실제 신청자 집단에 대한 주장을 뒷받침할 수 없습니다.
- 확률 보정, 비용 행렬, 심사 역량 제약, 공정성 분석, 시간 기준 검증은 포함하지 않습니다.
- Recall floor는 경제적으로 선택된 운영 정책이 아닌 민감도 분석 시나리오입니다.
