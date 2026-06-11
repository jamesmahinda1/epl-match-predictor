# ML2 Capstone Proposal: EPL Match Outcome Predictor

**Student:** James Mahinda

## Problem Statement

I want to predict whether a Premier League match ends in a home win, draw, or away win using historical match data from 1993 to 2023. The dataset has 12,026 matches across 31 seasons. The only available information per match is the date, teams, goals scored, and the result.

The challenge is that the raw data alone tells you nothing useful — you need to engineer features that capture each team's recent form before a match. I build rolling statistics from each team's last 5 games and use those to train classification models.

## Dataset

Premier League Matches 1993-2023 by Evan Gower (Kaggle). 12,026 matches, 50 teams, no missing values. Columns: Season_End_Year, Wk, Date, Home, HomeGoals, AwayGoals, Away, FTR.

## Approach

I start with exploratory analysis and SQL queries to understand the data, then build a full classification pipeline covering feature engineering, dimensionality reduction, multiple classifiers, hyperparameter tuning, and a neural network comparison.

**Feature engineering** — rolling 5-match form per team (goals scored, goals conceded, win rate, draw rate) computed without leaking future data. Log transforms on skewed features, StandardScaler for scaling, and PCA to reduce dimensions and visualize how well the classes separate.

**Classifiers** — LDA as a linear baseline, Random Forest, Gradient Boosting, and an LSTM that treats each team's last 10 results as a sequence input.

**Evaluation** — k-fold cross-validation, GridSearchCV tuning on the best model, final test set evaluation with confusion matrix and per-class F1. Draws are expected to be the hardest class since they are the least predictable outcome in football.

## Expected Outcomes

A model that beats the naive baseline of always predicting a home win (46% accuracy). The models will be ranked by macro F1 since the three classes are not balanced. The LSTM comparison will show whether sequence modeling adds anything over the aggregated form features.

## Limitations

The dataset has goals and results only — no shots, corners, lineups, or injury data. Everything is derived from past goals and results, which limits how much signal is available compared to what real prediction systems use.
