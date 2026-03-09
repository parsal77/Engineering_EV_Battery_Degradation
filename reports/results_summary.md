# Results Summary

## Benchmark Table

| Model | Task | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Random Forest Regressor | RUL | 5.0915 | 6.3732 | 0.9624 |
| XGBoost Regressor | RUL | 4.9354 | 6.3859 | 0.9623 |
| Transformer Encoder | RUL | 5.8665 | 6.5838 | 0.9519 |
| LSTM Neural Network | RUL | 29.4249 | 33.4632 | -0.2439 |
| Linear Regression | SOH | 0.0998 | 0.1160 | 0.9998 |
| XGBoost Regressor | SOH | 0.3677 | 0.5655 | 0.9945 |
| Random Forest Regressor | SOH | 0.4084 | 0.7148 | 0.9913 |
| Ridge Regression | SOH | 0.8853 | 0.9438 | 0.9847 |
| CNN-BiLSTM | SOH | 6.1839 | 6.8946 | 0.0135 |
| LSTM Neural Network | SOH | 6.2399 | 6.9417 | -0.0000 |

## Baseline Comparison

- SOH baseline (`Linear Regression`): MAE 0.0998, RMSE 0.1160, R² 0.9998
- RUL baseline (`Random Forest Regressor`): MAE 5.0915, RMSE 6.3732, R² 0.9624
- Tree-based models remained the most reliable for cross-battery RUL generalization in this experiment.