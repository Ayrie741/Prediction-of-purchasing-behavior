# Prediction-of-purchasing-behavior
Predict which visitors are **most likely to buy** and act on those insights.

---

## 1. Overview
- **Synthetic e‑commerce dataset** (5 000 rows) generated in‑notebook.  
- Notebook **`Prediction of purchasing behavior.ipynb`** walks through  
  *data → EDA → feature engineering → logistic model → evaluation → deployment*.

---

## 2. Data schema

| Column | Distribution | Reason |
| ------- | ----------- | ------ |
| `age` | Normal (μ 40, σ 15) clipped 18–80 | typical age curve |
| `time_spent_on_site` | Exponential (mean 5 min) | long‑tail sessions |
| `pages_visited` | Poisson (λ 5) | click‑depth |
| `previous_purchases` | Poisson 1 (+ boost if returner) | loyalty |
| `cart_value` | Exponential (scale 60) | basket skew |
| `is_returning_customer` | Binomial (p 0.35) | repeat visits |
| `days_since_last_visit` | Exponential (scale 25) | recency |

A custom formula turns these features into a probability and then the binary **`purchased`** label.

---

## 3. EDA highlights
- Users **< 30 yrs** & sessions **> 3 min** convert more.  
- `is_returning_customer` ↔ `previous_purchases` correlation ≈ 0.6.  
- `days_since_last_visit` is the strongest negative driver.

---

## 4. Feature engineering
- `recency_score = 1 / (1 + days_since_last_visit)`
- `engagement_score = time_spent_on_site × pages_visited / 10`
- `returning_x_engagement` interaction
- `log_cart_value` (log‑scaled)
- age buckets → one‑hot

---

## 5. Model pipeline
```text
ColumnTransformer → StandardScaler / OneHot → LogisticRegression
```  

- **80 / 20 train-test split**, **5-fold Stratified CV** (accuracy ≈ 0.72 ± 0.02)  
- **Test set** → **AUC** 0.76 | **Accuracy** 0.74

| Top feature                 | Odds ratio | Business insight             |
| --------------------------- | ---------- | ---------------------------- |
| `is_returning_customer`     | 1.75×      | retention is key             |
| `cart_value`                | 1.43×      | high basket → push checkout  |
| `days_since_last_visit`     | 0.69×/day  | recall within 30 days        |

## 6. Predicting new users

```python
from src.predict import predict_purchase_probability
import joblib, pandas as pd

# Load trained pipeline
pipe = joblib.load("models/purchase_prediction_model.pkl")

# read new visitor data
new_users = pd.read_csv("data/new_visitors.csv")

# score and preview
scored = predict_purchase_probability(new_users, pipe, proba_threshold=0.5)
print(scored.head())
```
## 7. Business applications

| Use-case         | Rule                            | Benefit                                 |
| ---------------- | ------------------------------- | --------------------------------------- |
| Targeted ads     | p ≥ 0.7                         | focus budget on high-intent users       |
| Dynamic pricing  | 0.4 ≤ p < 0.7                   | small discounts convert fence-sitters   |
| Personalised UX  | high p → premium<br>low p → trust content | lift engagement            |
| Inventory        | Σ p per SKU                     | demand forecast, avoid stockouts        |
| Sales forecast   | mean(p) × traffic               | plan revenue & budget                   |

---

## 8. Run locally

```bash
pip install -r requirements.txt
jupyter lab "notebooks/Prediction of purchasing behavior.ipynb"

