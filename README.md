# EPL Match Outcome Predictor

Predicting English Premier League match outcomes (home win, draw, or away win) from historical data, 1993 to 2023. Covers exploratory data analysis, SQL, feature engineering with ELO ratings, classification models, hyperparameter tuning, and time series forecasting across 12,026 matches and 31 seasons.

## Dataset

Premier League Matches 1993-2023 by Evan Gower (Kaggle). Columns: Season_End_Year, Wk, Date, Home, HomeGoals, AwayGoals, Away, FTR.

## Structure

- `notebooks/epl_match_predictor.ipynb` - main notebook
- `data/raw/` - raw match CSV (not tracked in git)
- `data/processed/` - processed features and SQLite database (not tracked in git)
- `PROPOSAL.md` - project proposal

## Methods

- Feature engineering: rolling 5-match form per team plus ELO team strength ratings
- Classification: Logistic Regression, Random Forest, Gradient Boosting, with GridSearchCV tuning
- Time series forecasting: ARIMA and LSTM on a team's scoring form

## Author

James Mahinda
