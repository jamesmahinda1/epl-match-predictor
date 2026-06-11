# ML2 Capstone Proposal: EPL Match Outcome Predictor

**Student:** James Mahinda

## Problem Statement

I want to predict whether a Premier League match ends in a home win, draw, or away win using historical match data from 1993 to 2023. The dataset has 12,026 matches across 31 seasons. The only information per match is the date, the two teams, the goals each scored, and the result.

The challenge is that the raw data on its own tells you very little. The work is in engineering features that capture each team's recent form and overall strength before a match, then using those to train classification models.

## Dataset

Premier League Matches 1993-2023 by Evan Gower (Kaggle). 12,026 matches, 50 teams, no missing values. Columns: Season_End_Year, Wk, Date, Home, HomeGoals, AwayGoals, Away, FTR.

## Approach

I start with exploratory analysis and SQL queries to understand the data, then build a classification pipeline, and finally reframe the same data as a time series forecasting problem.

Feature engineering: rolling 5-match form per team (goals scored, goals conceded, win rate, draw rate) computed without leaking future data, difference features comparing the home and away team's current form, and an ELO team strength rating. I check the feature distributions and scale them with StandardScaler before modeling.

Classification: Logistic Regression as a baseline, then Random Forest and Gradient Boosting, with GridSearchCV tuning. Models are evaluated with k-fold cross-validation, accuracy, macro F1, a confusion matrix, and learning curves to check for overfitting.

Time series forecasting: I take one team's scoring form (a rolling average of goals) and forecast it with ARIMA and an LSTM, comparing them on RMSE and MAE.

## Expected Outcomes

A classification model that beats the naive baseline of always predicting a home win (around 45 percent accuracy). The models are ranked by both accuracy and macro F1 since the three classes are not balanced, and I pick a final model based on which generalizes best rather than on raw accuracy alone. The forecasting section shows whether a neural network adds anything over a classical model on a single smooth series.

## Limitations

The dataset has goals and results only, no shots, corners, lineups, or injury data. Everything is derived from past goals and results, which limits how much signal is available compared to what real prediction systems use.
