# 신용 위험 모델링 파이프라인

## 프로젝트 소개

10,000건의 가상 금융 데이터를 직접 생성하고, 규칙 기반 베이스라인과 머신러닝 모델을 정량적으로 비교하는 지도학습 프로젝트입니다. 동일한 여섯 고객 특성으로 연체 위험을 분류하고 신용 점수를 회귀 예측하며, 데이터 누수 방지·불균형 처리·L1/L2 규제·앙상블 튜닝의 효과를 재현 가능한 실행 흐름으로 확인합니다.

## Project Scope

이 파이프라인은 연체 위험을 추정하여 대출 심사 의사결정을 지원하는 risk screening model입니다. 고위험 신청자의 추가 심사 우선순위를 지원할 수 있지만, 실제 고객을 자동 승인·거절하지 않으며 최종 결정은 정책과 추가 정보에 결합됩니다.

## 핵심 특징

- 여섯 조건의 규칙 기반 모델과 Logistic Regression, Random Forest 비교
- `class_weight="balanced"`를 이용한 불균형 학습과 5-fold 교차검증
- GridSearchCV로 12개 Random Forest 조합 최적화
- Ridge/Lasso의 다섯 Alpha별 계수 변화와 RMSE·MAE·R² 비교
- Confusion Matrix, ROC Curve, Feature Importance, 규제 계수 경로 생성
- 8:2 Train/Test 분할, `random_state=42`, Pipeline 기반 누수 방지

## 아키텍처

```mermaid
flowchart LR
    A[data_gen.py] --> B[finance_data.csv]
    B --> C[credit_risk.data]
    C --> D[credit_risk.preprocessing]
    D --> E[credit_risk.classification]
    D --> F[credit_risk.regression]
    E --> G[credit_risk.workflow]
    F --> G
    G --> H[credit_risk.reporting]
    H --> I[metrics.json·CSV·시각화]
    J[train.py CLI] --> G
```

`credit_risk` 패키지는 데이터 로드·분할, 공통 전처리, 분류와 회귀 학습, 결과 저장을 책임별 모듈로 분리합니다. 계산 함수인 `train_classification()`과 `train_regression()`은 파일을 생성하지 않아 독립적으로 재사용할 수 있고, `workflow`가 계산과 `reporting`을 연결합니다. 루트 `train.py`는 기존 실행 방법과 import 호환성을 유지하는 CLI 진입점입니다.

## 빠른 시작

Python 3.10 이상이 필요합니다.

`uv`를 사용하는 경우:

```bash
uv sync
uv run python data_gen.py
uv run python train.py
uv run python -m pytest -v
```

`pip`을 사용하는 경우:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_gen.py
python train.py
```

테스트는 다음 명령으로 실행합니다.

```bash
python -m pytest -v
```

`requirements.txt`와 `pyproject.toml`에는 프로젝트가 직접 사용하는 라이브러리만 명시합니다. 재현 가능한 전이 의존성 버전은 `uv.lock`에서 관리합니다.

`finance_data.csv`와 예측 CSV는 `.gitignore`의 `*.csv` 규칙으로 저장소에서 제외됩니다. 전체 실행 결과는 `artifacts/`에 생성됩니다.

## Data & Targets

입력 특성은 `age`, `annual_income`, `spending_score`, `debt_ratio`, `credit_card_count`, `overdue_count_6m`입니다. 제공된 생성 코드의 수식과 노이즈를 변경하지 않았습니다.

| 데이터 | 정상(0) | 연체(1) | 양성 비율 |
|---|---:|---:|---:|
| 전체 | 8,799 | 1,201 | 12.01% |
| Train | 7,039 | 961 | 12.01% |
| Test | 1,760 | 240 | 12.00% |

분류는 `test_size=0.2`, `stratify=y`, `random_state=42`로 분할하여 Train/Test의 연체 비율을 보존했습니다. 회귀도 동일한 8:2 비율과 시드를 사용합니다.

- Regression target: `credit_score`
- Classification target: `is_overdue`

## Class Imbalance & Classification Performance

| 모델 | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 예측 지연시간(ms) |
|---|---:|---:|---:|---:|---:|---:|
| 규칙 기반 | 0.7725 | 0.3285 | 0.8583 | 0.4752 | 0.8096 | 14.3352 |
| Logistic Regression | 0.8785 | 0.4965 | **0.8750** | **0.6335** | **0.9505** | **2.7185** |
| Random Forest | **0.9095** | **0.6528** | 0.5250 | 0.5820 | 0.9376 | 44.4808 |
| Random Forest (튜닝) | 0.8840 | 0.5102 | 0.8333 | 0.6329 | 0.9409 | 56.2024 |

Logistic Regression은 규칙 기반보다 F1-Score가 약 33.3%, ROC-AUC가 약 17.4% 향상되었습니다. Accuracy만 보면 기본 Random Forest가 가장 높지만 연체 Recall이 0.5250으로 낮아, 연체 고객을 정상으로 놓치는 비용이 큰 신용 심사에는 부적합합니다.

튜닝된 Random Forest의 최적값은 `max_depth=8`, `min_samples_split=5`, `n_estimators=200`이며 Train 내부 CV ROC-AUC는 0.9468입니다. GridSearchCV는 risk score의 순위 구분 능력(ROC-AUC)으로 모델을 고르고, 운영 임계값은 별도 Train 내부 CV에서 Recall 90% 이상을 만족하는 범위에서 Precision을 최대화해 선택합니다.

![Confusion Matrix](artifacts/confusion_matrix.png)

![ROC Curve](artifacts/roc_curve.png)

### 평가지표 선정 근거

전체 양성 비율이 12.01%이므로 Accuracy는 다수 클래스의 영향을 크게 받습니다. 따라서 Accuracy를 단독으로 사용하지 않고, 양성 클래스의 Precision과 Recall을 함께 반영하는 F1-Score와 임계값 전반의 순위 구분 능력을 나타내는 ROC-AUC를 핵심 지표로 사용했습니다.

`class_weight="balanced"`로 학습했으므로 `predict_proba()` 값은 검증된 실제 연체 확률이 아니라 model-estimated overdue probability 또는 risk score로 해석합니다. 확률 보정(calibration)은 이번 범위에 포함하지 않았습니다.

규칙 기반 모델은 여섯 `if` 조건의 최종 0/1 판단만 반환합니다. 이 값으로 AUC 계산은 가능하지만 ROC가 하나의 거친 운영점에 기반하므로, `predict_proba()`가 반환한 연속 위험 확률로 계산한 ML 모델의 AUC와 동일한 수준으로 해석하면 안 됩니다.

### 규칙 기반 베이스라인

다음 조건 중 하나라도 만족하면 연체 위험으로 분류합니다.

1. 최근 6개월 연체가 2회 이상
2. 부채 비율이 80% 초과이고 연 소득이 4,500만원 미만
3. 연 소득이 2,500만원 미만
4. 소비 점수가 90 초과이고 부채 비율이 70% 초과
5. 카드가 8개 이상이고 부채 비율이 65% 초과
6. 25세 미만이고 부채 비율이 75% 초과

규칙은 설명과 변경이 쉽지만 경계가 불연속적이고 특성 간 복잡한 결합을 학습하지 못합니다. ML은 데이터에서 가중치와 상호작용을 학습해 F1/AUC를 높였지만, 데이터 품질과 분포 변화에 의존하고 정기적인 검증이 필요합니다.

### 불균형 처리

SMOTE 대신 Logistic Regression과 Random Forest에 `class_weight="balanced"`를 사용했습니다. 데이터 자체를 합성하지 않고 학습 손실에서 소수 클래스의 중요도를 높이며 Pipeline을 단순하게 유지할 수 있기 때문입니다.

| 방법 | 장점 | 한계 |
|---|---|---|
| Class weight | 합성 샘플 없이 비용 민감 학습, Pipeline이 단순함 | 모델이 가중치 기능을 지원해야 하며 확률 보정이 달라질 수 있음 |
| SMOTE | 소수 클래스의 학습 표본을 직접 보강 | 부적절한 합성점·노이즈 증폭 가능, CV 내부 리샘플링 구성이 필요함 |

`class_weight`는 학습 과정의 손실 계산에만 영향을 주며 Test Set 자체의 분포는 변경하지 않습니다. SMOTE를 사용하더라도 Train/CV fold 내부에서만 적용해야 하며, Test Set은 실제 분포를 보존해야 공정한 일반화 성능을 평가할 수 있습니다.

## Hyperparameter Optimization

Random Forest는 `n_estimators`(100, 200), `max_depth`(None, 8, 16), `min_samples_split`(2, 5)의 12개 조합을 `GridSearchCV(scoring="roc_auc")`로 비교합니다. 이 단계는 연속 risk score를 잘 생성하는 모델을 고르는 과정이며, Test Set은 사용하지 않습니다.

## Decision Threshold

모든 ML 모델은 기본 정책 `threshold=0.5`를 유지합니다. F1 최대 threshold는 참고값으로 보고하며, 최종 운영 threshold는 같은 Train Set 내부 CV에서 Recall 90% 이상을 만족하는 후보 중 Precision이 가장 높은 값으로 선택합니다. Test Set은 임계값이나 hyperparameter 선택에 전혀 사용하지 않고, 선택이 끝난 정책의 일반화 성능을 한 번 평가하는 데만 사용합니다. ROC-AUC는 연속 score로 계산하므로 threshold가 바뀌어도 동일합니다.

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5000 | 0.4965 | 0.8750 | 0.6335 | 0.9505 |
| Logistic Regression | 0.7576 (F1 reference) | **0.6310** | 0.7625 | **0.6906** | **0.9505** |
| Logistic Regression | 0.4848 (Recall ≥ 0.90 CV) | 0.4919 | 0.8833 | 0.6319 | 0.9505 |
| Tuned Random Forest | 0.5000 | 0.5102 | **0.8333** | 0.6329 | 0.9409 |
| Tuned Random Forest | 0.5186 (F1 reference) | 0.5197 | 0.8250 | 0.6377 | 0.9409 |
| Tuned Random Forest | 0.3523 (Recall ≥ 0.90 CV) | 0.4563 | **0.8917** | 0.6037 | 0.9409 |

Logistic Regression의 Test Set 오류 건수는 F1만으로 운영 threshold를 정하면 왜 위험해질 수 있는지 보여줍니다.

| Logistic policy | FN (위험 고객 누락) | FP (정상 고객 추가 심사) |
|---|---:|---:|
| Default 0.5 | 30 | 213 |
| F1 reference | 57 | 107 |
| Recall ≥ 0.90 CV | 28 | 219 |

F1-optimal은 F1을 0.6335에서 0.6906으로 높였지만 FN도 30건에서 57건으로 거의 두 배가 됩니다. 신용리스크 screening에서는 이 미탐 증가가 업무 목적과 맞지 않을 수 있으므로, F1 결과는 정책 선택의 참고값으로만 사용합니다. 최종 운영 후보는 Recall 제약을 먼저 적용해 FN을 28건으로 제한하고, 그 범위에서 Precision이 가장 높은 Logistic Regression threshold를 사용합니다. 그 대가로 정상 고객 추가 심사(FP)는 기본값보다 6건 늘어납니다.

![Threshold comparison confusion matrices](artifacts/confusion_matrix_threshold_comparison.png)

![Tuned Random Forest threshold trade-off](artifacts/threshold_tradeoff.png)

## FP / FN Interpretation

- False Positive: 실제 정상 고객을 고위험으로 분류하는 오류입니다. 불필요한 추가 심사, 승인 지연, 우량 고객 이탈 또는 영업 기회비용으로 이어질 수 있습니다.
- False Negative: 실제 연체 고객을 정상으로 분류하는 오류입니다. 위험 대출 승인과 연체·신용손실 증가 가능성을 높입니다.

실제 금융기관은 두 오류의 경제적 비용, risk appetite, 심사 인력, 정책을 함께 반영해 임계값을 정합니다. 이 데이터에는 실제 손실액이 없으므로 `FN cost = 5 × FP cost` 같은 임의 금전 가정은 하지 않았습니다. 대신 위험고객 누락을 제한하기 위해 Train Set 내부 CV의 Recall 90% 제약을 먼저 적용하고, 그 안에서 Precision이 가장 높은 threshold를 운영 정책으로 선택했습니다.

### 앙상블과 편향·분산

Random Forest는 여러 결정트리의 예측을 평균하여 단일 트리의 높은 분산을 낮춥니다. 이번 실험에서도 깊이를 제한한 튜닝 모델이 기본 Forest보다 Recall과 F1을 높여 과도하게 세분화된 경계를 완화했습니다. 다만 데이터 생성 신호가 소득·부채·연체 횟수의 선형 관계에 가까워 Logistic Regression이 튜닝 Forest보다 F1, AUC, 지연시간 모두 우수했습니다. 앙상블이 항상 선형 모델보다 낫지는 않으며 데이터의 구조와 운영 비용을 함께 봐야 합니다.

### Feature Importance 해석

![Feature Importance](artifacts/feature_importance.png)

- 연 소득의 중요도가 50.9%로 가장 높아 분류 경계 형성에 가장 많이 사용되었습니다.
- 최근 연체 횟수 24.5%, 부채 비율 16.2%가 뒤를 이어 데이터 생성 규칙과 일관된 결과를 보였습니다.
- 수치형 카드 수의 중요도는 1.7%로 예측 기여가 제한적이었습니다.
- 중요도는 영향의 방향이나 인과관계를 의미하지 않으며, 분할 기반 중요도에는 편향이 있을 수 있습니다.

## 회귀 성능

Alpha 선택은 Train Set 내부 5-fold CV 평균 RMSE만으로 수행했습니다. 선택된 모델을 전체 Train Set에 다시 학습한 뒤 Test Set은 최종 평가에 한 번만 사용했습니다.

| 모델 | 선택 Alpha | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| Ridge | 1.0 | 29.3667 | 23.2739 | 0.8607 |
| Lasso | 0.1 | **29.3599** | **23.2711** | **0.8607** |

두 모델 모두 약 86%의 신용점수 분산을 설명했으며 Lasso가 근소하게 우수했습니다. 지표는 clipping 전 원시 예측값으로 계산하여 범위 밖 오차를 숨기지 않았고, 사용자에게 제공하는 `credit_score_predictions.csv`의 예측값만 0~1000으로 제한했습니다.

![Ridge와 Lasso Alpha별 계수 변화](artifacts/regularization_coefficients.png)

그래프의 계수는 원래 단위가 아니라 전처리 후 표준화 공간 기준입니다. Ridge의 L2 규제는 계수를 연속적으로 축소하면서 대부분 남기는 반면, Lasso의 L1 규제는 Alpha가 커질수록 중요도가 낮은 계수를 정확히 0으로 만들어 변수 선택 효과를 냅니다. 실제로 Lasso Alpha 1에서는 카드 수를 포함한 영향이 작은 특성의 계수가 0이 되었고, Alpha 100에서는 모든 계수가 0으로 수축해 CV RMSE가 80.25까지 악화되었습니다.

## 데이터 누수 방지

- `credit_score`는 `is_overdue` 생성에 직접 사용되므로 분류 입력에서 제외했습니다. 두 타깃 모두 여섯 입력 특성에 포함하지 않았습니다.
- 여섯 수치형 특성의 중앙값 대치와 Standard Scaling을 `ColumnTransformer`와 모델 `Pipeline` 안에 배치했습니다.
- 전처리기는 각 Train/CV fold에서만 fit되고 Test Set에는 transform만 수행합니다.
- Random Forest GridSearchCV와 Ridge/Lasso Alpha 선택은 Train 내부 5-fold 결과만 사용합니다.
- Test Set은 하이퍼파라미터 선택이 끝난 모델의 일반화 성능 산출에만 사용합니다.

원 데이터에는 명시적 범주형 열이 없습니다. `credit_card_count`와 `overdue_count_6m`은 정수로 표현된 횟수형 특성이므로, 값의 순서와 크기 정보를 보존하도록 다른 네 특성과 함께 수치형으로 처리했습니다.

## 운영 모델과 임계값

최종 운영 후보는 Recall-constrained Logistic Regression입니다. Tuned RF보다 운영 정책 F1(0.6319 vs. 0.6037)과 ROC-AUC(0.9505 vs. 0.9409)가 높고, Test batch prediction latency도 2.7185ms로 56.2024ms보다 훨씬 낮습니다. 선형 모델은 구조와 설명도 더 단순합니다. Holdout Test의 Recall은 각각 0.8833과 0.8917로 CV의 90% 제약과 차이가 날 수 있으므로, 실제 운영 전에는 독립 검증 기간에서도 이 기준을 재확인해야 합니다.

## 산출물

| 파일 | 역할 |
|---|---|
| `data_gen.py` | 제공된 규칙으로 10,000건 데이터 생성 |
| `train.py` | 기존 API를 제공하고 전체 분석을 실행하는 CLI 진입점 |
| `credit_risk/data.py` | 데이터 스키마, 로드·검증, Train/Test 분할 |
| `credit_risk/preprocessing.py` | 여섯 수치형 특성의 공통 전처리 Pipeline 생성 |
| `credit_risk/classification.py` | 규칙·Logistic·Random Forest 학습 및 평가 |
| `credit_risk/regression.py` | Ridge·Lasso 선택, 학습 및 평가 |
| `credit_risk/reporting.py` | 그래프와 CSV·JSON 산출물 저장 |
| `credit_risk/workflow.py` | 분류·회귀 계산과 리포팅 실행 흐름 조립 |
| `tests/test_data_gen.py` | 데이터 크기·범위·재현성 검증 |
| `tests/test_train.py` | 누수 방지, 규칙, 분류·회귀·산출물 검증 |
| `artifacts/metrics.json` | 클래스 분포, CV, Test 성능, 계수·중요도 기록 |
| `artifacts/confusion_matrix.png` | Logistic/튜닝 Forest 오분류 비교 |
| `artifacts/confusion_matrix_threshold_comparison.png` | 기본·CV tuned threshold의 TP/FP/FN/TN 비교 |
| `artifacts/roc_curve.png` | 규칙·ML 모델 ROC-AUC 비교 |
| `artifacts/threshold_tradeoff.png` | Tuned Random Forest의 threshold별 Precision/Recall/F1 |
| `artifacts/feature_importance.png` | 튜닝 Forest 특성 중요도 |
| `artifacts/regularization_coefficients.png` | Ridge/Lasso Alpha별 계수 변화 |

## 한계

- 가상 데이터의 생성 규칙을 학습한 결과이므로 실제 고객의 신용 심사에 사용할 수 없습니다.
- 실제 대출 손실액이 없어 FP/FN의 금전 비용 기반 임계값을 산출하지 않았습니다.
- probability calibration을 검증하지 않았습니다.
- 실제 금융기관의 승인 정책이나 규제 요건을 반영한 production model이 아닙니다.
- 공정성, 설명 가능성, 시간에 따른 데이터 드리프트는 이 미션 범위에 포함하지 않았습니다.
- 모델 지연시간은 현재 장비의 2,000건 Test Set 배치 예측 기준이며 온라인 단건 지연시간과 다릅니다.
