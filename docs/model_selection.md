# 모델 선택 및 실험 기록

## 실험 질문

이 프로젝트는 단순 규칙 기반 모델, 선형 분류기, 랜덤 포레스트 중 어떤 모델이 가상 연체 사례를 가장 잘 구분하는지, 규제가 가상 신용점수 회귀에 어떤 영향을 주는지, 그리고 임계값과 트리 수 선택에 따라 어떤 상충 관계가 나타나는지를 확인합니다. 실제로 운영 가능한 신용 정책을 주장하지는 않습니다.

## 평가 프로토콜

- `data_gen.py`는 10,000개 행을 결정적으로 생성합니다. 분류는 클래스 비율을 보존하는 80:20 분할을 사용하고, 회귀도 `random_state=42`로 동일한 분할 비율을 사용합니다.
- 특성 전처리는 각 scikit-learn `Pipeline` 내부에 있습니다. 따라서 결측치 대치와 스케일링은 교차검증 중에는 각 학습 폴드에서만 fit되고, holdout 추론 전에는 전체 학습 데이터에서만 fit됩니다.
- 랜덤 포레스트 파라미터는 학습 데이터 내부 5-fold `GridSearchCV(scoring="roc_auc")`로 선택합니다. 보류 검증 세트의 레이블은 모델·하이퍼파라미터·임계값 선택에는 사용하지 않습니다. 보류 검증 세트는 선택이 완료된 모델과 정책의 최종 성능 평가 및 ROC 곡선, 혼동 행렬 같은 평가 시각화에만 사용합니다.
- 리지와 라쏘의 alpha 후보(`0.01`, `0.1`, `1`, `10`, `100`)는 학습 데이터 내부 5-fold 평균 RMSE로 선택한 뒤 holdout에서 한 번 평가합니다.

## 분류 결과

커밋된 [지표 기록](../artifacts/metrics.json)이 수치의 기준 기록입니다. 현재의 결정적 가상 데이터에서 로지스틱 회귀는 holdout ROC-AUC가 가장 높고, Tuned RF는 F1이 조금 높습니다. 기본 모델 선택은 holdout 결과가 아니라 학습 5-fold CV ROC-AUC와 단순성을 기준으로 합니다. 이는 가상 결과가 대체로 선형적인 소득·부채·연체 횟수 신호에 의해 생성되기 때문이며, 일반적으로 선형 모델을 선호해야 한다는 뜻은 아닙니다.

규칙 기반 모델은 해석 가능한 기준선으로 유지합니다. 이진 출력으로도 AUC를 구할 수 있지만, 하나의 거친 운영점에 기반하므로 연속 위험 점수를 내는 모델의 ROC 곡선과 직접 같은 수준으로 비교할 수는 없습니다.

## 임계값 민감도 분석

운영 threshold는 자동으로 선택하지 않습니다. 선택된 Logistic Regression의 Train OOF probability만으로 `np.linspace(0.30, 0.60, 31)`을 평가하고, `0.50`을 baseline으로 표시한 Precision·Recall·F1·TP·FP·FN 표를 만듭니다. Holdout 레이블은 threshold sweep이나 선택에 절대 사용하지 않습니다.

손실 추정치, 심사 역량, 위험 선호도가 없으므로 최적 threshold를 주장할 수 없습니다. 표와 그래프는 사람이 비용과 운영 제약을 반영해 threshold를 선택하기 위한 근거이며, 선택 후에만 holdout으로 일반화를 평가합니다.

## 랜덤 포레스트 트리 수 민감도 분석

먼저 `GridSearchCV`로 랜덤 포레스트의 하이퍼파라미터를 선택합니다. 이후 트리 수 민감도 분석에서는 선택된 `max_depth`와 `min_samples_split`을 고정하고, `n_estimators`만 25·50·100·200·300·500으로 변경해 동일한 학습 5-fold 교차검증에서 비교합니다. 이는 선택된 트리 수 이외 설정을 유지한 채 트리 수의 영향을 분리합니다. 하이퍼파라미터 선택 뒤 학습 교차검증을 재사용하므로 이 결과는 편향 없는 선택 후 성능 추정치가 아닌 기술적 민감도 분석입니다. 이 한계를 문서에 남기는 조건이라면 중첩 교차검증은 이 포트폴리오 프로젝트에는 과도합니다.

이 데이터에서는 약 200개 이후 평균 ROC-AUC 개선폭이 작아지는 모습을 보입니다. 그래프의 폴드 표준편차 음영은 폴드 간 변동성이지 트리 수 차이의 유의성 검정이 아닙니다. `GridSearchCV` 결과가 모델 선택의 근거이고, 포화 구간 그래프는 그 선택의 실용적인 비용·성능 맥락을 설명합니다.

## 예측 시간 해석

예측 시간은 간단한 재현성 보조 정보이지 성능 측정 주장이 아닙니다. 모든 분류 모델과 트리 수별 예측 시간은 1회 warm-up 뒤 5회 반복하여, 동일한 최대 2,000행 학습 특성 배치에서 측정합니다. 정확한 값은 `metrics.json`의 단일 `benchmark` 구역에 기록하며, 하드웨어, CPU 부하, 라이브러리 버전, `n_jobs`에 따라 달라질 수 있습니다. 온라인 단건 예측 시간과 비교해서는 안 됩니다.

## 산출물

- `artifacts/metrics.json`: 기계가 읽을 수 있는 지표, 선택된 파라미터, 회귀 기록, 단일 성능 측정 구역
- `artifacts/confusion_matrix.png`: 0.50 baseline의 FP/FN 비교
- `artifacts/threshold_sweep.csv`, `threshold_sweep.png`: Logistic Train OOF threshold 후보 표와 그래프
- `artifacts/roc_curve.png`: 분류 성능 비교
- `artifacts/random_forest_n_estimators_curve.png`, `feature_importance.png`: 앙상블 선택·해석 근거
- `artifacts/regularization_coefficients.png`: Ridge/Lasso 규제 강도별 계수 변화
- Git에서 제외되는 `classification_predictions.csv`, `credit_score_predictions.csv`: 로컬에서 재생성 가능한 행 단위 예측 결과

## 선택 요약

| 결정 항목 | 선택 | 근거 |
| --- | --- | --- |
| 분류 모델 | 로지스틱 회귀 | Train 5-fold CV ROC-AUC가 Tuned RF보다 높고 모델이 단순함. Holdout 결과는 선택 후 일반화 성능 확인에 사용 |
| 랜덤 포레스트 트리 수 | 200 | CV ROC-AUC가 약 200개 이후 포화 |
| 임계값 정책 | 사람 검토용 Train OOF sweep | 실제 비용 정보가 없어 단일 최적 정책을 자동 선택하지 않음 |
| baseline | 0.50 | `0.30~0.60` 후보와 TP·FP·FN 및 Precision·Recall·F1을 비교 |
| 리지 alpha | 1.0 | 학습 5-fold CV RMSE |
| 라쏘 alpha | 0.1 | 학습 5-fold CV RMSE |
