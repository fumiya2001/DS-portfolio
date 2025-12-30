## Overview
This notebook summarizes EDA and model training for the Titanic survival prediction task using Kaggle dataset.

## What I Focused On

### Exploratory Data Analysis (EDA)
- Checked data distributions
- Analyzed missing values
- Explored relationships between each feature and survival

### Feature Engineering
- Filled missing values
- Created new features (`Sex_Pclass`, `log_Fare`, `familysize`, `IsChild`, `Title`)

### Modeling
- Trained machine learning models from scikit-learn and Gradient Boosting
- Compared model performance to select the best approach
- Performed hyperparameter tuning using Optuna

### Result
- Created submission file based on the trained model 
- Achieved an accuracy of **0.78468** on the public leaderboard (Kaggle)