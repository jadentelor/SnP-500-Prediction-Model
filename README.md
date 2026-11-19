# Predicting S&P 500 Returns from Interest Rates

DATA1902 group project, University of Sydney (2026).

## Background

We tested whether changes in the US federal funds rate (FFR) help predict next-month S&P 500 returns. Two machine learning models, Lasso regression and gradient boosting, were compared against a simple baseline that always predicts the average month.

## Data

- S&P 500 daily prices, 2001-2023: Yahoo Finance (^GSPC), converted to monthly returns
- Effective federal funds rate (FEDFUNDS), monthly: FRED

## Method

- Features: lagged FFR levels and FFR changes (1 to 3 months), plus the 3-month change and its direction. Each month only uses FFR data from before that month.
- Training: 2001-2018, with time-series cross-validation
- Testing: 2019-2023 (out of sample)
- Models: baseline (average month), Lasso, gradient boosting

## Results (test period 2019-2023)

| Model | MAE (pp) | RMSE (pp) | Direction accuracy |
| --- | --- | --- | --- |
| Baseline (average month) | 4.45 | 5.54 | 65% |
| Lasso | 4.63 | 5.87 | 55% |
| Gradient boosting | 4.54 | 5.63 | 55% |

Neither model beat the baseline out of sample. Investing $10,000 only in months the model predicted "up" also finished behind always staying invested ($13,706 Lasso and $12,527 gradient boosting vs $17,921). Monthly returns look very hard to predict from interest rates alone.

## How to run

From the main folder:

```
pip install pandas numpy matplotlib scikit-learn
python stage2_model/stage2_model.py
```

Results and figures are saved to `stage2_model/outputs/`.

## Folders

- `data_cleaned/`: cleaned S&P 500 data
- `data_original/`: federal funds rate data (FRED)
- `stage2_model/`: model script and outputs
