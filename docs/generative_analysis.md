# Synthetic Data 생성 구조 분석

이 문서는 이 프로젝트의 가상 데이터가 **어떤 수학적 구조로 생성되었는지**, 그리고 학습된 모델이 그 구조를 **얼마나 잘 다시 찾아냈는지**를 설명합니다.

이 분석은 [`data_gen.py`](../data_gen.py)에 이미 공개된 생성 공식을 이용한 **사후 진단(post-hoc diagnostic)** 입니다.

생성 공식을 모델 학습, 하이퍼파라미터 선택, threshold 선택에 사용하지 않았으며, 아래 oracle 결과도 새로운 모델 선택 근거로 사용하지 않습니다.

---

## 1. 신용점수는 어떻게 만들어지는가

`data_gen.py`에서 가상의 `credit_score`는 다음 구조로 생성됩니다.

$$
S
=
300
+0.03I
-50O
-100D
+\varepsilon
$$

여기서

- $I$: `annual_income`
- $O$: `overdue_count_6m`
- $D$: `debt_ratio`
- $\varepsilon \sim \mathcal{N}(0,30^2)$: 평균 0, 표준편차 30의 랜덤 노이즈

입니다.

노이즈를 제외한 평균 신용점수를 $\mu(X)$라고 하면

$$
\mu(X)
=
300+0.03I-50O-100D
$$

입니다.

즉 데이터 생성자가 신용점수를 만들 때 직접 사용한 feature는 세 개입니다.

| Feature | 생성식의 계수 | 의미 |
| --- | ---: | --- |
| `annual_income` | +0.03 | 소득이 높을수록 신용점수 상승 |
| `overdue_count_6m` | -50 | 최근 연체가 많을수록 신용점수 하락 |
| `debt_ratio` | -100 | 부채비율이 높을수록 신용점수 하락 |
| `age` | 0 | 생성식에 직접 사용되지 않음 |
| `spending_score` | 0 | 생성식에 직접 사용되지 않음 |
| `credit_card_count` | 0 | 생성식에 직접 사용되지 않음 |

같은 `random_state=42`로 데이터를 재현하면 `credit_score`의 15% 분위수는

$$
q_{0.15}=292
$$

입니다.

그리고 `is_overdue=1`은 다음 조건으로 생성됩니다.

1. `credit_score < 292`
2. 그중 독립적인 난수 조건을 통과한 약 80%

따라서 분류 target에는 **구조적으로 설명 가능한 위험 신호**와 **의도적으로 추가된 랜덤성**이 함께 존재합니다.

---

## 2. 분류 문제의 실제 위험 구조 유도

### 2.1 고객의 연체 위험은 무엇으로 결정되는가

반올림 전 신용점수를 $Z$라고 두겠습니다.

$$
Z
=
\mu(X)+\varepsilon
$$

이고

$$
\varepsilon
\sim
\mathcal{N}(0,30^2)
$$

이므로, feature $X$가 주어졌을 때

$$
Z\mid X
\sim
\mathcal{N}(\mu(X),30^2)
$$

입니다.

실제 코드에서는 신용점수를 정수로 반올림한 뒤

```python
credit_score < 292
```

를 검사합니다.

연속값 $Z$가 정확히 `.5`가 될 확률은 사실상 0이므로, 이 조건은 거의 확실하게

$$
Z<291.5
$$

와 같습니다.

따라서 고객이 저신용점수 영역에 들어갈 확률은

$$
P(Z<291.5\mid X)
=
\Phi
\left(
\frac{291.5-\mu(X)}{30}
\right)
$$

입니다.

여기서 $\Phi$는 표준정규분포의 누적분포함수(CDF)입니다.

저신용점수 영역에 들어간 사람 중 약 80%에게 `is_overdue=1`이 부여되므로, threshold를 고정해서 해석하면 조건부 연체확률은 대략

$$
P(Y=1\mid X)
\approx
0.8
\Phi
\left(
\frac{291.5-\mu(X)}{30}
\right)
$$

가 됩니다.

이제

$$
\mu(X)
=
300+0.03I-50O-100D
$$

를 대입하면

$$
P(Y=1\mid X)
\approx
0.8
\Phi
\left(
\frac{
291.5-
\left(
300+0.03I-50O-100D
\right)
}{30}
\right)
$$

입니다.

괄호 안을 정리하면

$$
291.5-300-0.03I+50O+100D
$$

이므로, 고객마다 달라지는 핵심 부분은

$$
\boxed{
R(X)
=
-0.03I+50O+100D
}
$$

입니다.

즉 이 synthetic dataset에서 고객의 **이론적 위험 순위**는 사실상

- 소득이 낮을수록 위험 증가
- 최근 연체가 많을수록 위험 증가
- 부채비율이 높을수록 위험 증가

라는 하나의 선형 점수로 정리됩니다.

---

## 3. 왜 복잡한 확률식 대신 선형 위험점수만 봐도 되는가

연체확률에는

$$
\Phi(\cdot)
$$

라는 정규분포 CDF가 들어갑니다.

하지만 $\Phi$는 **단조 증가 함수**입니다.

즉

$$
R(X_a)>R(X_b)
$$

라면 반드시

$$
P(Y=1\mid X_a)
>
P(Y=1\mid X_b)
$$

입니다.

따라서 위험확률의 정확한 숫자는 달라도 고객의 순서는

$$
R(X)
=
-0.03I+50O+100D
$$

만으로 결정할 수 있습니다.

ROC-AUC는 확률값 자체가 `0.73`인지 `0.84`인지보다

> 실제 양성 사례를 실제 음성 사례보다 더 높은 위험점수에 배치했는가?

를 보는 ranking metric입니다.

따라서 이 프로젝트에서는

$$
\boxed{
R(X)
=
-0.03I+50O+100D
}
$$

를 데이터 생성 구조를 알고 있을 때 얻을 수 있는 **generative oracle risk score**로 사용할 수 있습니다.

이 oracle은 실제 서비스용 모델이 아닙니다.

생성식을 이미 알고 있기 때문에 계산할 수 있는 **진단용 기준점**입니다.

---

## 4. Logistic Regression은 실제 위험 구조를 찾아냈는가

현재 Logistic Regression은 `class_weight="balanced"`와 선택된 `C=0.01`을 사용합니다.

전처리 과정에서 모든 feature에 `StandardScaler`가 적용되므로, Logistic Regression 내부의 coefficient는 원래 단위가 아니라 표준화된 단위입니다.

표준화가

$$
z_j
=
\frac{x_j-\mu_j}{\sigma_j}
$$

이고 모델이

$$
\beta_j^{scaled}z_j
$$

를 사용한다면

$$
\beta_j^{scaled}
\frac{x_j-\mu_j}{\sigma_j}
$$

로 쓸 수 있습니다.

따라서 원래 feature 단위에서의 coefficient는

$$
\boxed{
\beta_j^{raw}
=
\frac{\beta_j^{scaled}}{\sigma_j}
}
$$

가 됩니다.

### 4.1 Logistic coefficient와 생성식 coefficient 비교

Logistic Regression은 생성 과정에서 사용된 정규분포 CDF가 아니라 logistic sigmoid를 사용합니다.

즉 생성 모델은 대략

$$
P(Y=1\mid X)
\propto
\Phi(a+b^TX)
$$

이고 Logistic Regression은

$$
P(Y=1\mid X)
\approx
\sigma(c+w^TX)
$$

입니다.

따라서 두 모델의 coefficient **절대 크기**를 그대로 비교하면 안 됩니다.

대신 한 coefficient를 기준으로 전체를 같은 비율로 rescale하면 **방향과 상대적인 비율**을 비교할 수 있습니다.

`debt_ratio` coefficient를 생성식과 동일하게 `100`으로 맞춰 비교하면 다음과 같습니다.

| Feature | 이론적 risk 계수 | Logistic 상대 계수 |
| --- | ---: | ---: |
| `annual_income` | -0.0300 | -0.0283 |
| `overdue_count_6m` | +50.0 | +47.34 |
| `debt_ratio` | +100.0 | +100.0 |

실제 생성식은

$$
-0.03I+50O+100D
$$

였고 Logistic Regression은 거의 같은 상대적 방향을 학습했습니다.

생성식에 직접 들어가지 않았던 feature의 상대 coefficient는 다음과 같습니다.

| Feature | Logistic 상대 계수 |
| --- | ---: |
| `age` | -0.0226 |
| `spending_score` | +0.0273 |
| `credit_card_count` | +0.3191 |

feature마다 단위가 다르기 때문에 이 숫자를 그대로 feature importance처럼 비교해서는 안 됩니다.

여기서 중요한 결과는 **실제 생성에 사용된 세 변수의 방향과 상대적인 비율이 거의 그대로 복원되었다는 점**입니다.

---

## 5. Logistic과 Oracle은 실제로 고객을 같은 순서로 정렬하는가

coefficient가 비슷하더라도 실제 예측 순서까지 같은지는 별도로 확인할 수 있습니다.

holdout 2,000개 샘플에 대해

1. 생성식에서 유도한 oracle score

$$
R(X)
=
-0.03I+50O+100D
$$

2. Logistic Regression의 `predict_proba()`

를 각각 계산했습니다.

그리고 두 점수의 **Spearman rank correlation**을 측정했습니다.

Spearman correlation은 두 값의 절대적인 크기보다

> 두 방법이 관측값들을 얼마나 비슷한 순서로 정렬하는가?

를 측정합니다.

결과는

$$
\boxed{
\rho_{Spearman}
=
0.99955
}
$$

였습니다.

Spearman correlation은 `1`에 가까울수록 두 ranking이 거의 같다는 뜻입니다.

따라서

$$
0.99955\approx1
$$

이라는 결과는

> Logistic Regression과 생성식에서 직접 유도한 위험점수가 고객들을 거의 같은 위험 순서로 배치하고 있다.

는 뜻입니다.

즉 Logistic Regression의 높은 ROC-AUC는 특정 hyperparameter 값을 우연히 잘 골라서 나타난 결과라기보다, **실제 데이터 생성에 사용된 위험 방향을 모델이 거의 그대로 학습했다는 해석**과 잘 맞습니다.

---

## 6. Oracle ROC-AUC와 Logistic ROC-AUC 비교

같은 holdout에서 두 위험점수의 ROC-AUC를 계산하면 다음과 같습니다.

| Score | Holdout ROC-AUC |
| --- | ---: |
| Generative oracle risk score | **0.950181** |
| Logistic Regression | **0.950526** |
| Logistic - Oracle | **+0.000344** |

차이는 약

$$
3.44\times10^{-4}
$$

입니다.

사실상 매우 작은 차이입니다.

여기서 한 가지 의문이 생깁니다.

> 생성 공식을 알고 만든 oracle인데 왜 Logistic Regression의 관측 AUC가 아주 조금 더 높은가?

이것은 모순이 아닙니다.

`is_overdue` 생성 과정에는 다음 난수 조건이 들어갑니다.

```python
np.random.rand(N_SAMPLES) > 0.2
```

즉 신용점수가 충분히 낮은 고객이라도 약 20%는 `is_overdue=0`으로 남습니다.

따라서 target에는 feature만으로는 예측할 수 없는 랜덤성이 존재합니다.

또 holdout은 모집단 전체가 아니라 2,000개의 유한 표본입니다.

생성 구조상 최적인 ranking이더라도 특정 유한 표본에서 반드시 가장 높은 관측 AUC를 가져야 하는 것은 아닙니다.

따라서 여기서 중요한 것은

$$
0.950181
\quad\text{vs}\quad
0.950526
$$

의 아주 작은 순위 차이가 아니라,

- 두 AUC가 사실상 같은 수준이며
- 두 risk score의 Spearman correlation이 `0.99955`

라는 점입니다.

---

## 7. 이것이 Logistic `C` 실험을 어떻게 설명하는가

현재 Logistic Regression의 `C` sensitivity 결과는 다음과 같습니다.

| C | CV ROC-AUC mean |
| ---: | ---: |
| 0.001 | 0.953075 |
| 0.003 | 0.953720 |
| 0.01 | **0.953895** |
| 0.03 | 0.953857 |
| 0.1 | 0.953855 |

`C=0.01`에서 가장 높은 평균 CV ROC-AUC를 기록했습니다.

하지만

$$
C=0.01,\;0.03,\;0.1
$$

의 차이는 매우 작습니다.

예를 들어

$$
0.953895-0.953855
=
0.000040
$$

정도입니다.

따라서 이 결과를

> `C=0.01`이라는 특별한 숫자가 모델 성능을 크게 향상시켰다.

라고 해석하는 것은 적절하지 않습니다.

앞의 oracle 분석과 연결하면 더 자연스러운 설명이 가능합니다.

1. Logistic Regression이 이미 실제 생성 위험 방향을 거의 정확하게 찾았습니다.
2. `C`는 regularization의 강도를 바꾸므로 coefficient 크기와 score scale에는 영향을 줄 수 있습니다.
3. 하지만 적절한 범위에서는 coefficient의 **방향과 고객 ranking**이 크게 변하지 않습니다.
4. ROC-AUC는 ranking metric이므로 결과 역시 거의 변하지 않습니다.

즉

> **이 데이터에서 `C`가 의미 없는 것이 아니라, 실제 위험구조가 강하고 단순하기 때문에 Logistic의 ranking 성능이 넓은 regularization 범위에서 안정적이다.**

라고 해석할 수 있습니다.

---

# 8. Ridge와 Lasso는 신용점수 생성식을 복원했는가

분류보다 회귀에서는 생성 구조를 더 직접적으로 비교할 수 있습니다.

노이즈를 제외한 실제 신용점수 생성식은

$$
S^*
=
300
+0.03I
-50O
-100D
$$

입니다.

즉 실제 구조적 coefficient를 이미 알고 있습니다.

| Feature | True coefficient |
| --- | ---: |
| Intercept | 300 |
| `annual_income` | +0.03 |
| `overdue_count_6m` | -50 |
| `debt_ratio` | -100 |
| `age` | 0 |
| `spending_score` | 0 |
| `credit_card_count` | 0 |

현재 프로젝트에서는 CV RMSE를 기준으로

- Ridge: `alpha=1.0`
- Lasso: `alpha=0.1`

이 선택되었습니다.

---

## 9. StandardScaler의 coefficient를 원래 단위로 되돌리기

회귀 모델에도 `StandardScaler`가 적용되기 때문에 모델에 저장된 coefficient를 생성식과 바로 비교할 수 없습니다.

표준화된 변수

$$
z_j
=
\frac{x_j-\mu_j}{\sigma_j}
$$

를 사용하는 회귀식이

$$
\hat y
=
\beta_0^{scaled}
+
\sum_j
\beta_j^{scaled}z_j
$$

라고 하겠습니다.

$z_j$를 대입하면

$$
\hat y
=
\beta_0^{scaled}
+
\sum_j
\beta_j^{scaled}
\frac{x_j-\mu_j}{\sigma_j}
$$

입니다.

이를 풀어 쓰면

$$
\hat y
=
\beta_0^{scaled}
+
\sum_j
\frac{\beta_j^{scaled}}{\sigma_j}x_j
-
\sum_j
\beta_j^{scaled}
\frac{\mu_j}{\sigma_j}
$$

입니다.

따라서 원래 단위의 coefficient는

$$
\boxed{
\beta_j^{raw}
=
\frac{\beta_j^{scaled}}{\sigma_j}
}
$$

이고 원래 단위의 intercept는

$$
\boxed{
\beta_0^{raw}
=
\beta_0^{scaled}
-
\sum_j
\beta_j^{scaled}
\frac{\mu_j}{\sigma_j}
}
$$

가 됩니다.

이 변환을 적용하면 Ridge와 Lasso가 실제 데이터 생성식을 얼마나 복원했는지 직접 비교할 수 있습니다.

---

## 10. 실제 생성 coefficient와 Ridge/Lasso 비교

결과는 다음과 같습니다.

| Feature | True | Ridge | Lasso |
| --- | ---: | ---: | ---: |
| Intercept | 300.000 | 299.758 | 299.456 |
| `annual_income` | +0.0300 | +0.02998 | +0.02994 |
| `overdue_count_6m` | -50.000 | -50.316 | -50.184 |
| `debt_ratio` | -100.000 | -99.950 | -99.606 |
| `age` | 0 | -0.00768 | -0.00085 |
| `spending_score` | 0 | -0.01110 | -0.00752 |
| `credit_card_count` | 0 | +0.12614 | +0.08732 |

실제 신호 세 개만 보면 거의 원래 계수를 복원했습니다.

### `annual_income`

True:

$$
0.03
$$

Ridge:

$$
0.0299817
$$

상대 오차는

$$
\frac{|0.0299817-0.03|}{0.03}
\times100
\approx
0.06\%
$$

입니다.

Lasso:

$$
0.0299368
$$

상대 오차는 약

$$
0.21\%
$$

입니다.

### `overdue_count_6m`

True:

$$
-50
$$

Ridge:

$$
-50.3155
$$

상대 오차는 약

$$
0.63\%
$$

입니다.

Lasso:

$$
-50.1843
$$

상대 오차는 약

$$
0.37\%
$$

입니다.

### `debt_ratio`

True:

$$
-100
$$

Ridge:

$$
-99.9504
$$

상대 오차는 약

$$
0.05\%
$$

입니다.

Lasso:

$$
-99.6064
$$

상대 오차는 약

$$
0.39\%
$$

입니다.

정리하면:

| Feature | Ridge relative error | Lasso relative error |
| --- | ---: | ---: |
| `annual_income` | 0.06% | 0.21% |
| `overdue_count_6m` | 0.63% | 0.37% |
| `debt_ratio` | 0.05% | 0.39% |

주요 세 coefficient 모두 오차가 1% 미만입니다.

따라서 Ridge와 Lasso 모두 **실제 데이터 생성 평균식을 매우 가깝게 복원**했다고 볼 수 있습니다.

생성식에 사용되지 않은

- `age`
- `spending_score`
- `credit_card_count`

의 coefficient도 0 주변의 작은 값으로 학습되었습니다.

Lasso가 이 세 coefficient를 정확히 0으로 만들지는 않았으므로

> Lasso가 irrelevant feature를 완전히 제거했다.

고 주장하지는 않습니다.

현재 선택된 `alpha=0.1`에서는 불필요한 변수들의 영향을 작은 값으로 억제했다고 해석하는 것이 적절합니다.

---

# 11. 왜 회귀 RMSE는 약 30 아래로 내려가기 어려운가

신용점수에는 처음부터 다음 random noise가 추가됩니다.

$$
\varepsilon
\sim
\mathcal{N}(0,30^2)
$$

표준편차는

$$
\sigma=30
$$

입니다.

feature $X$만 알고 있는 모델은 특정 고객에게 실제로 더해진 랜덤한 $\varepsilon$을 알 수 없습니다.

예를 들어 두 고객의 모든 feature가 완전히 같아도 각각 다른 noise가 추가될 수 있습니다.

따라서 모델이 생성식의 평균 부분

$$
300+0.03I-50O-100D
$$

을 완벽하게 알고 있어도 개별 target과는 noise만큼 차이가 날 수 있습니다.

노이즈만 남았다고 가정하면 평균 squared error는

$$
E[\varepsilon^2]
=
\sigma^2
=
30^2
=
900
$$

입니다.

따라서 이에 대응하는 RMSE 규모는

$$
\sqrt{E[\varepsilon^2]}
=
\sqrt{900}
=
30
$$

입니다.

즉 이 데이터에서는 대략

$$
\boxed{
RMSE\approx30
}
$$

부근에 **noise floor**가 존재할 것으로 예상할 수 있습니다.

---

## 12. 실제 Oracle RMSE 계산

실제 생성 평균식

$$
\hat S_{oracle}
=
300
+0.03I
-50O
-100D
$$

을 그대로 holdout 예측값으로 사용했습니다.

이 oracle은 모델을 학습한 것이 아닙니다.

데이터 생성자가 사용한 평균 공식을 그대로 넣은 것입니다.

결과는

$$
\boxed{
RMSE_{oracle}
=
29.32995
}
$$

였습니다.

이론적 noise scale인 `30`과 매우 가깝습니다.

정확히 30이 아닌 이유는

- holdout이 무한 모집단이 아니라 유한한 2,000개 표본이고
- 실제 `credit_score`에는 clipping과 rounding이 적용되기 때문입니다.

현재 Ridge/Lasso와 비교하면 다음과 같습니다.

| Model | Holdout RMSE | Oracle 대비 차이 |
| --- | ---: | ---: |
| Generative oracle mean | **29.32995** | - |
| Lasso | 29.35993 | +0.02999 |
| Ridge | 29.36666 | +0.03671 |

Lasso와 oracle의 차이는

$$
29.35993-29.32995
\approx
0.02999
$$

입니다.

oracle RMSE 대비 약

$$
\frac{0.02999}{29.32995}\times100
\approx
0.10\%
$$

입니다.

Ridge 역시

$$
29.36666-29.32995
\approx
0.03671
$$

로, 약

$$
0.13\%
$$

차이입니다.

즉 Ridge와 Lasso는 단순히 “RMSE가 29 정도 나온 모델”이 아니라

> **데이터 생성 평균식을 알고 있는 oracle의 예측오차에 거의 도달한 모델**

이라고 해석할 수 있습니다.

---

# 13. 그렇다면 HPO의 개선폭이 작았던 이유는 무엇인가

이제 지금까지의 실험 결과를 하나의 구조로 연결할 수 있습니다.

## Classification

실제 조건부 위험은 대략

$$
P(Y=1\mid X)
\approx
0.8\Phi(a+b^TX)
$$

형태입니다.

즉 **선형 risk score에 단조 함수를 적용한 구조**입니다.

Logistic Regression도

$$
P(Y=1\mid X)
=
\sigma(c+w^TX)
$$

형태입니다.

둘의 link function은 다릅니다.

생성 구조:

$$
\Phi(\cdot)
$$

Logistic Regression:

$$
\sigma(\cdot)
$$

하지만 둘 다 단조 증가 함수입니다.

ROC-AUC에서는 확률값 자체보다 ranking이 중요하므로, Logistic Regression이 올바른 선형 방향 $w$만 찾으면 generative oracle과 거의 같은 ranking을 만들 수 있습니다.

실제로

$$
\rho_{Spearman}=0.99955
$$

였습니다.

따라서 Logistic Regression은 이미 데이터 생성 위험구조와 거의 같은 고객 순위를 만들어냈습니다.

이 상태에서는 `C`를 더 세밀하게 조절해도 ranking을 크게 개선할 여지가 작습니다.

---

## Regression

회귀에서는 실제 평균식 자체가

$$
300
+0.03I
-50O
-100D
$$

라는 선형식입니다.

Ridge와 Lasso 역시 선형 모델입니다.

따라서 충분한 데이터가 있다면 실제 coefficient를 직접 복원하기 좋은 문제입니다.

실제로 주요 coefficient의 상대 오차는 모두 1% 미만이었고,

$$
RMSE_{oracle}=29.32995
$$

에 대해

$$
RMSE_{Lasso}=29.35993
$$

$$
RMSE_{Ridge}=29.36666
$$

로 거의 noise floor에 도달했습니다.

따라서 더 복잡한 모델이나 더 세밀한 HPO를 적용하더라도 얻을 수 있는 추가 개선폭 자체가 제한적입니다.

---

# 14. 이 프로젝트에서 얻는 더 중요한 결론

처음에는 다음 질문에 집중할 수 있습니다.

> 어떤 `C`가 가장 좋은가?

> `min_samples_split`은 20인가 40인가?

> 트리는 200개인가 300개인가?

이 질문들도 모델 선택 과정에서는 필요합니다.

하지만 생성 구조까지 분석하면 더 중요한 질문으로 넘어갈 수 있습니다.

> 왜 hyperparameter를 바꿔도 결과가 크게 변하지 않는가?

이 프로젝트에서는 그 이유를 다음과 같이 설명할 수 있습니다.

### 1. 문제 자체의 핵심 구조가 비교적 단순하다

분류의 실제 위험 순위는

$$
-0.03I+50O+100D
$$

라는 선형점수로 표현됩니다.

### 2. Logistic Regression이 그 위험 방향을 거의 그대로 학습했다

Oracle과 Logistic의 Spearman correlation:

$$
0.99955
$$

### 3. 따라서 regularization을 조금 바꿔도 ranking이 거의 유지된다

그래서 `C` 변화에 비해 ROC-AUC 변화가 매우 작습니다.

### 4. 회귀 역시 실제 생성 평균이 선형이다

Ridge와 Lasso가 실제 coefficient를 1% 미만의 오차로 복원했습니다.

### 5. target 자체에 표준편차 30의 랜덤 noise가 있다

따라서 평균식을 완벽하게 알아도 RMSE를 0으로 만들 수 없습니다.

실제 oracle RMSE가

$$
29.32995
$$

였습니다.

### 6. 현재 Ridge/Lasso는 이미 그 oracle에 매우 가깝다

추가 모델 복잡도나 HPO가 개선할 수 있는 공간이 작습니다.

따라서 이 프로젝트에서 중요한 결과는

> **최고의 하이퍼파라미터 숫자를 아주 세밀하게 찾았다는 것보다, 모델 선택 결과가 넓은 설정에서 안정적이며 그 이유를 데이터 생성 구조로 설명할 수 있다는 것**

입니다.

---

# 15. Synthetic dataset이기 때문에 가능한 분석

이 분석은 실제 금융 데이터에서는 일반적으로 할 수 없습니다.

실제 데이터에서는 세상의 진짜 조건부 확률

$$
P(Y\mid X)
$$

이나 실제 데이터 생성 coefficient를 알 수 없기 때문입니다.

예를 들어 실제 고객 데이터에서 Logistic Regression이

$$
w_1,w_2,w_3,\dots
$$

를 학습했다고 해도

> 이 coefficient가 세상의 진짜 coefficient와 얼마나 가까운가?

를 직접 확인할 방법은 없습니다.

하지만 이 프로젝트의 데이터는 직접 생성했기 때문에

$$
\beta_{true}
$$

를 알고 있습니다.

따라서 단순히

> 모델의 예측 성능이 좋다.

에서 끝나지 않고

> 모델이 알려진 생성 구조를 얼마나 정확하게 다시 찾아냈는가?

까지 확인할 수 있습니다.

즉 synthetic dataset의 한계를 숨기기보다, 오히려 그것을 이용해 **model recovery와 noise floor를 검증하는 실험 환경**으로 사용할 수 있습니다.

---

# 16. 해석상의 한계

이 분석에는 다음 한계가 있습니다.

1. **Oracle은 실제 배포 모델이 아닙니다.**  
   생성 공식을 이미 알고 있기 때문에 계산 가능한 진단용 기준입니다.

2. **Oracle 결과를 보고 모델을 다시 선택해서는 안 됩니다.**  
   이 분석은 기존 Train/CV 기반 모델 선택이 끝난 뒤 결과를 설명하기 위한 post-hoc analysis입니다.

3. **실제 금융 데이터의 구조가 선형이라는 의미가 아닙니다.**  
   이 결과는 현재 synthetic generator의 구조에 한정됩니다.

4. **`is_overdue`에는 의도적인 랜덤성이 존재합니다.**  
   따라서 feature만으로 모든 target을 완벽하게 예측할 수 없습니다.

5. **약 12%의 positive rate는 실제 금융기관의 연체율이나 PD를 의미하지 않습니다.**

6. **Logistic Regression의 `predict_proba()` 역시 실제 보정된 PD로 해석하지 않습니다.**  
   이 프로젝트에서는 가상 데이터에 대한 model risk score입니다.

7. **실제 운영 비용, 공정성, 시간에 따른 분포 변화, 규제 적합성은 이 분석에서 검증하지 않습니다.**

---

# 17. 결론

이 synthetic dataset에서는 모델 성능 숫자뿐 아니라 **모델이 알려진 데이터 생성 구조를 얼마나 복원했는지**까지 확인할 수 있습니다.

핵심 결과는 다음 세 가지입니다.

### Classification: 실제 위험구조 복원

데이터 생성식에서 유도되는 위험 ranking은

$$
\boxed{
R(X)
=
-0.03I+50O+100D
}
$$

입니다.

Logistic Regression은 이 세 변수의 방향과 상대적인 크기를 거의 동일하게 학습했습니다.

Oracle risk score와 Logistic score의 holdout Spearman correlation은

$$
\boxed{
0.99955
}
$$

였습니다.

또한 holdout ROC-AUC는

$$
AUC_{oracle}
=
0.950181
$$

$$
AUC_{logistic}
=
0.950526
$$

로 사실상 같은 수준이었습니다.

따라서 Logistic Regression의 높은 ROC-AUC와 `C` 변화에 대한 안정성은 데이터 생성 구조와 연결해 설명할 수 있습니다.

---

### Regression: 실제 coefficient 복원

실제 신용점수 평균식은

$$
\boxed{
S^*
=
300+0.03I-50O-100D
}
$$

입니다.

Ridge와 Lasso는 주요 세 coefficient를 모두 1% 미만의 상대 오차로 복원했습니다.

즉 단순히 예측값만 잘 맞힌 것이 아니라, 데이터를 만든 실제 선형 구조까지 매우 가깝게 다시 찾아냈습니다.

---

### Noise floor: 더 이상 크게 좋아지기 어려운 이유

신용점수에는

$$
\varepsilon
\sim
\mathcal{N}(0,30^2)
$$

의 랜덤 noise가 존재합니다.

생성 평균식을 직접 사용하는 oracle의 holdout RMSE는

$$
\boxed{
29.32995
}
$$

였습니다.

현재 모델은

$$
RMSE_{Lasso}
=
29.35993
$$

$$
RMSE_{Ridge}
=
29.36666
$$

으로 oracle과 약 `0.03` 정도밖에 차이가 나지 않았습니다.

따라서 이 프로젝트에서 추가 HPO나 모델 복잡도 증가의 개선폭이 작았던 것은 단순한 탐색 실패가 아니라,

> **현재 모델들이 이미 데이터 생성 구조와 이 데이터가 허용하는 예측 한계에 매우 가까워졌기 때문**

이라고 설명할 수 있습니다.