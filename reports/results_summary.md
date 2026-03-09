# Results Summary

_Single source of truth: `results/metrics.csv`._

## Benchmark Table

| Model | Task | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Random Forest Regressor | RUL | 5.0915 | 6.3732 | 0.9624 |
| XGBoost Regressor | RUL | 4.9354 | 6.3859 | 0.9623 |
| Transformer Encoder | RUL | 7.9898 | 9.1193 | 0.9076 |
| LSTM Neural Network | RUL | 29.6019 | 33.7142 | -0.2626 |
| Linear Regression | SOH | 0.0998 | 0.1160 | 0.9998 |
| XGBoost Regressor | SOH | 0.3677 | 0.5655 | 0.9945 |
| Random Forest Regressor | SOH | 0.4084 | 0.7148 | 0.9913 |
| Ridge Regression | SOH | 0.8853 | 0.9438 | 0.9847 |
| CNN-BiLSTM | SOH | 1.6264 | 1.9431 | 0.9216 |
| LSTM Neural Network | SOH | 6.2334 | 6.9414 | 0.0000 |

## Baseline Comparison

- SOH baseline (`Linear Regression`): MAE 0.0998, RMSE 0.1160, R² 0.9998
- RUL baseline (`Random Forest Regressor`): MAE 5.0915, RMSE 6.3732, R² 0.9624
- This report is auto-generated from `results/metrics.csv` to keep documentation and metrics synchronized.