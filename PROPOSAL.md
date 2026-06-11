# ML2 Capstone Proposal: EPL Match Outcome Predictor

**Student:** James Mahinda

## Problem Statement

I want to predict the outcome of English Premier League matches (home win, draw, or away win) using historical match data from 1993 to 2023. This is a 3-class classification problem. The data has 12,026 matches across 31 seasons with goals and results for every game.

The naive baseline is always predicting a home win, which is correct about 46% of the time. My goal is to build a model that beats that baseline using engineered form features and advanced classification techniques.

## Dataset

Premier League Matches 1993-2023 by Evan Gower (Kaggle). Columns: Season_End_Year, Wk, Date, Home, HomeGoals, AwayGoals, Away, FTR. 12,026 matches, 50 teams, no missing values.

## Approach

**Feature Engineering**
- Rolling 5-match form per team: goals scored, goals conceded, win rate, draw rate
- Form difference features: home team advantage vs away team in each stat
- Log transformation on skewed features
- StandardScaler for feature scaling
- PCA to reduce dimensionality and visualize class separation

**Modeling**
- LDA (Linear Discriminant Analysis) as a baseline classifier
- Random Forest
- Gradient Boosting
- Model comparison using accuracy, macro F1, and confusion matrix

**Tuning**
- K-fold cross-validation for robust evaluation
- GridSearchCV hyperparameter tuning on the best performing model

## ML2 Topics Covered

| Week | Topics Used |
|---|---|
| W1 | Feature engineering, log transforms, StandardScaler, PCA, LDA, cross-validation |
| W2 | Clustering teams by playing style as supporting analysis |
| W3 | Time-aware rolling features (no data leakage across match dates) |
| W4 | LSTM/RNN discussed as future improvement in conclusion |

## Expected Outcomes

A classification model that beats the 46% home-win baseline. The best model will be identified through cross-validation and tuned with GridSearchCV. Final evaluation on a held-out test set with confusion matrix and per-class F1 scores showing where the model struggles most (draws are the hardest class to predict in football).

## Limitations

The dataset only has goals and results, not shots, corners, player lineups, or injury data. All features are derived from past goals and results only. Real betting models use much richer data.
