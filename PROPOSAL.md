# Final Project Proposal: EPL Match Outcome Predictor

**Student:** James Mahinda

## Problem Statement

I want to predict English Premier League match outcomes using historical match data from 1993 to 2023. Specifically, I will predict whether a home team wins, draws, or loses, and separately predict the total number of goals scored. Football prediction is a well-defined supervised learning problem where recent form, head-to-head history, and home advantage all influence results.

## Dataset

Premier League Matches 1993-2023 by Evan Gower, downloaded from Kaggle. Contains the date, home team, away team, home goals, away goals, and full-time result (H/D/A) for every EPL match across 31 seasons. 12,026 matches total, 50 unique teams, no missing values.

## Objectives

1. Explore the data and identify key patterns (home advantage, seasonal trends, top team dominance over time)
2. Store the match data in a SQLite database and run analytical SQL queries
3. Engineer rolling form features that capture team performance without leaking future data
4. Predict match outcome (classification: H/D/A) using multiple ML models
5. Predict total goals in a match (regression)
6. Cluster teams by playing style based on their seasonal goal statistics
7. Analyze season-by-season points trajectories for selected teams using time series

## Techniques Used

| Module | Technique |
|---|---|
| Python / pandas | Data loading, cleaning, feature engineering |
| Data Analysis | EDA, distributions, home advantage analysis, team trends |
| SQL | SQLite tables, aggregate queries, decade-by-decade comparisons |
| ML1 - Classification | Logistic Regression and Random Forest to predict H/D/A |
| ML1 - Regression | Random Forest to predict total goals per match |
| ML1 - Clustering | K-Means to group teams by attacking and defensive style |
| Advanced ML2 | Rolling 5-match form features, hyperparameter tuning, time series points trend |

## Expected Outcomes

A working end-to-end notebook covering all Zindua modules in one project. The classifier is expected to reach around 52-55% accuracy on a 3-class problem (random baseline is 33%, always-predict-home baseline is around 46%). The regression model will be evaluated on RMSE and R2. Feature importance plots will show which rolling stats drive predictions most.

## Limitations

The dataset contains only goals and match results, not shots, corners, or cards. Feature engineering is therefore based on form (goals scored, goals conceded, win rate) rather than in-game intensity metrics. Player injuries, lineups, and transfer activity are also not included since that data is not in the source file.
