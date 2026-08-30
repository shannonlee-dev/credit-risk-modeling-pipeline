# Credit Risk Modeling Pipeline Design

## 목표와 범위

제공된 `data_gen.py` 로직으로 생성한 10,000건의 고객 데이터를 사용해 규칙 기반 분류, 머신러닝 분류, 규제 회귀, 앙상블 튜닝을 한 번의 재현 가능한 실행 흐름으로 제공한다. `mission.md`와 `rublic.md`의 필수 요구사항만 구현하며 Custom Transformer, 별도 패키지 모듈화, Learning Curve 같은 보너스 항목은 제외한다.

## 실행 구조

- `data_gen.py`는 제공된 난수 시드와 생성 규칙을 그대로 사용하여 `finance_data.csv`를 만든다.
- `train.py`는 데이터를 읽고 검증한 뒤 분류와 회귀 실험을 실행하며 결과표와 시각화 파일을 `artifacts/`에 저장한다.
- `tests/`는 데이터 생성, 규칙 모델, 전처리 구조, 누수 방지, 모델 평가 결과의 핵심 계약을 검증한다.
- `README.md`는 실행 방법, 실제 성능표, 그래프 해석, 데이터 누수와 평가지표 선택 근거를 설명한다.

두 개의 실행 스크립트만 두어 필수 실행 흐름은 단순하게 유지한다. 별도 애플리케이션 계층이나 Notebook은 만들지 않는다.

학습 스크립트 내부는 데이터 로드·검증, 공통 특성 정의, 데이터 분할, 전처리기 생성, 규칙 기반 평가, 분류 모델 학습·평가, 회귀 모델 학습·평가, 시각화·결과 저장 단계로 역할을 분리한다. 회귀와 분류는 동일한 입력 특성과 공통 전처리 원칙을 공유하되, 데이터 분할 이후 타깃·모델·평가지표에서 각각 독립적으로 분기한다.

## 데이터와 전처리

모델 입력은 `age`, `annual_income`, `spending_score`, `debt_ratio`, `credit_card_count`, `overdue_count_6m`의 여섯 변수다. 분류 타깃을 직접 생성하는 데 쓰인 `credit_score`와 타깃 `is_overdue`는 입력에서 제외해 타깃 누수를 방지한다.

분류 데이터는 `train_test_split(test_size=0.2, stratify=y, random_state=42)`로 나눈다. 회귀 데이터도 같은 8:2 비율과 시드를 사용한다. 데이터 생성 직후 전체 데이터와 분할 직후 Train/Test Set의 연체 클래스별 건수와 양성 비율을 산출해 stratified split이 실제 분포를 보존했는지 확인한다.

`credit_card_count`는 수치적 순서가 있지만 수가 적고 유한한 카드 보유 범주이므로 수치형·범주형 전처리 경로를 모두 검증하기 위한 실습 목적으로 범주형 변수로 취급한다. 최빈값 대치 후 One-Hot Encoding하고, 나머지 수치형 변수는 중앙값 대치 후 Standard Scaling한다. 이 선택과 한계를 README에 명시한다. 모든 전처리기는 Scikit-learn `ColumnTransformer`와 `Pipeline` 안에서 Train Set에만 fit한다.

## 분류 설계

규칙 기반 모델은 소득, 부채 비율, 최근 연체 횟수, 카드 수, 소비 점수, 나이를 활용한 여섯 개의 명시적 `if` 조건으로 위험 여부를 판단한다. Accuracy, Precision, Recall, F1-Score, ROC-AUC를 Test Set에서 계산한다. 이 모델은 연속 확률이 아닌 0/1 판단을 반환하므로 AUC와 ROC가 거친 단일 운영점에 기반하며, `predict_proba()` 기반 ML 모델의 연속 위험점수와 동일하게 해석하지 않는다.

머신러닝 모델은 다음 두 가지다.

- Logistic Regression: `class_weight="balanced"`로 불균형을 보정하고 5-fold 교차검증을 수행한다.
- Random Forest: `class_weight="balanced"`를 고정한 기본 하이퍼파라미터 모델의 성능을 먼저 측정한 뒤, 같은 가중치 설정에서 100개 이하 조합의 `GridSearchCV(cv=5)`로 `n_estimators`, `max_depth`, `min_samples_split`을 탐색한다. 기본 모델과 튜닝 모델의 성능 차이로 최적화 효과를 정량화한다.

불균형 처리는 SMOTE 대신 `class_weight="balanced"`를 사용한다. 데이터를 합성하지 않고 학습 시 손실 계산에서 소수 클래스의 중요도를 높이며 Pipeline을 단순하게 유지하기 위해서다. README에서 SMOTE와 class weight의 장단점을 비교하고, class weight는 학습 과정에만 영향을 주며 Test Set 자체의 분포는 변경하지 않는다고 명시한다.

Accuracy는 다수 클래스의 영향을 크게 받으므로 단독으로 사용하지 않는다. 양성 클래스 탐지 성능을 반영하는 F1-Score와 임계값 전반의 분류 능력을 평가하는 ROC-AUC를 핵심 지표로 함께 사용한다. ML 모델은 Test Set 확률로 ROC-AUC를 평가한다. 최종 산출물은 모델별 비교표, 예측 확률, Confusion Matrix, ROC Curve, Random Forest Feature Importance다.

Random Forest는 여러 결정트리의 예측을 결합해 단일 트리의 높은 분산을 줄이는 효과를 기대한다. README에서 규칙·선형 모델과 튜닝된 앙상블 모델의 실제 성능 차이를 편향-분산 관점에서 해석한다. 평가 임계값은 기본 0.5를 사용하되, 연체 고객을 정상으로 판단하는 미탐의 예상 손실이 더 크다고 가정해 Recall/F1을 우선하는 조정 방법을 설명한다. 모델별 예측 지연시간과 복잡도도 측정·비교하고, 성능 향상이 지연시간 증가를 정당화할 때만 앙상블을 최종 운영 모델로 선택한다.

## 회귀 설계

Ridge와 Lasso에 각각 `0.01`, `0.1`, `1`, `10`, `100`의 Alpha를 적용한다. 각 조합을 동일한 Train/Test 데이터와 전처리 파이프라인으로 학습해 RMSE, MAE, R²를 비교하고, 5-fold 교차검증의 평균 RMSE가 가장 낮은 Alpha를 모델별 대표값으로 선택한다.

모든 예측값은 금융 점수 범위에 맞게 0~1000으로 제한하여 내보낸다. 각 Alpha에서 전처리 후 특성명을 기준으로 표준화 공간의 모델 계수를 추출해 Ridge와 Lasso 계수 변화 그래프를 생성하고 L1/L2 차이를 README에서 실제 결과와 연결해 설명한다.

## 산출물과 오류 처리

실행 결과는 `artifacts/metrics.json`, 예측 샘플, PNG 시각화에 저장한다. CSV 데이터는 `.gitignore`의 `*.csv` 규칙으로 버전 관리에서 제외하며, 실제 성능 수치는 README 표에 기록한다. 입력 파일이 없으면 생성 명령을 안내하는 명확한 오류를 내고, 필수 열이 누락되면 누락된 열 이름과 함께 실행을 중단한다.

## 검증과 완료 기준

테스트는 생성 데이터의 크기·열·재현성, 전체 및 분할 데이터의 클래스 분포, 규칙 개수와 이진 출력, 전처리기의 수치/범주 경로, 입력 특성에서 타깃 제거, 평가 지표와 산출물 생성을 확인한다. 전체 테스트 통과 후 실제 10,000건 데이터로 `train.py`를 끝까지 실행하고 모든 성능표와 그래프가 생성되는지 확인한다.

완료 시 저장소에는 필수 구현, 실제 결과가 반영된 문서, 고정된 의존성 명세가 있으며 데이터 생성·모델링·문서화 단위의 Conventional Commit 이력이 남아 있어야 한다.
