# House Prices - Advanced Regression Techniques

A full regression modeling walkthrough on Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition — EDA, cleaning, feature engineering, categorical encoding, outlier diagnosis, model comparison, hyperparameter tuning, and a validated blended ensemble.

## Problem

Predict final sale prices for residential homes in Ames, Iowa from 79 explanatory variables (size, quality, location, condition, etc.). Evaluation metric: RMSE between the log of predicted and actual sale price.

## Approach

1. **EDA** — distribution of `SalePrice`, missing-value audit, correlation of numeric features with price
2. **Cleaning** — distinguished "NaN means the feature doesn't exist" (e.g. no pool, no garage) from genuinely missing data, and handled each appropriately
3. **Feature engineering** — combined related columns into stronger signals (`TotalSF`, `HouseAge`, `TotalBath`) and added binary "has this feature" flags
4. **Encoding** — ordinal mapping for quality/condition scales (preserving order), one-hot encoding for nominal categories
5. **Outlier diagnosis** — identified and removed two houses with implausible size-to-price ratios from training data only
6. **Modeling** — compared Ridge, Lasso, Random Forest, and XGBoost via 5-fold cross-validation
7. **Tuning** — `RandomizedSearchCV` on XGBoost (30 sampled combinations across 5 hyperparameters)
8. **Blending** — weighted average of Ridge + Lasso + XGBoost, with weights selected via cross-validation rather than guessed

## Results

| Stage | CV RMSE (log) |
|---|---|
| Baseline Ridge, with outliers | 0.1445 |
| Baseline Ridge, outliers removed | 0.1152 |
| Blend, untuned XGBoost, guessed weights | ~0.1113 |
| Blend, tuned XGBoost, searched weights | **0.1102** |

**Public leaderboard: 0.12803**

The gap between CV RMSE (0.1102) and public leaderboard RMSE (0.12803) is discussed directly in the notebook rather than treated as a discrepancy to ignore — likely contributors include the leaderboard's partial-test-set scoring, outlier removal only applying to training data, and CV variance from relatively small fold sizes (~290 rows/fold).

*Note: this competition's top leaderboard entries (scores near 0.0) have historically been associated with exploiting the publicly available Ames Housing source data rather than standard modeling. This project focuses on leakage-free validation and reproducible modeling instead of leaderboard position.*

## What I learned

- Two outliers accounted for a large share of the initial error — outlier diagnosis mattered more than model choice at that stage
- XGBoost underperformed Ridge/Lasso individually but still improved the blend meaningfully, suggesting it captured different error patterns rather than just being weaker
- Blending outperformed every individual model
- CV performance did not perfectly predict leaderboard performance, and that gap was worth quantifying rather than dismissing

## Repository contents

- `house-prices-walkthrough.ipynb` — full notebook
- `requirements.txt` — Python dependencies
- `LICENSE`

## Running locally

```bash
pip install -r requirements.txt
```

The notebook was written and run on Kaggle, where the competition data is mounted automatically at `/kaggle/input/house-prices-advanced-regression-techniques/`. To run it outside Kaggle:

1. Download `train.csv`, `test.csv`, and `data_description.txt` from the [competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) (requires a free Kaggle account and accepting the competition rules)
2. Place them in the same directory as the notebook
3. Update the two `pd.read_csv(...)` paths at the top of the notebook from `/kaggle/input/house-prices-advanced-regression-techniques/train.csv` to `train.csv` (and same for `test.csv`)

Easiest path overall: open the notebook directly on Kaggle (via the competition's **Code** tab → **New Notebook**), where the data is already attached and no path changes are needed.

## Possible next steps

- Target encoding for `Neighborhood`
- Interaction features (e.g. `OverallQual × TotalSF`)
- LightGBM/CatBoost as additional blend members
- A stacked meta-model instead of a fixed-weight blend
- Error analysis by price bucket and feature importance inspection

**Find the Notebook here:** [Kaggle Notebook](https://www.kaggle.com/code/hadiaaakhaaan/house-prices-prediction-competition-notebook)


## Author

Hadia Khan
[LinkedIn](https://linkedin.com/in/hadia-khan-478a83328) · [GitHub](https://github.com/Hadia-Khan-version)
