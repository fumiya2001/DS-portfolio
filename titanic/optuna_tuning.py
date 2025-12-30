import optuna
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def optimize_lgb(X, y, n_trials=100, seed=42):
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        gbm = lgb.LGBMClassifier(**param, random_state=seed)
        
        score = cross_val_score(gbm, X, y, n_jobs=-1, cv=5).mean()
        return score
    sampler = optuna.samplers.TPESampler(seed=42)
   
    study = optuna.create_study(direction='maximize',sampler=sampler) 
    study.optimize(objective, n_trials=n_trials) 

    return study.best_params


def optimize_svc(X, y, n_trials=60):
    def objective_svc(trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e1, log=True),
        }

        model = SVC(**params, probability=True,random_state=42)
        
        score = cross_val_score(model, X, y, n_jobs=-1, cv=5).mean()
        return score
    
    sampler = optuna.samplers.TPESampler(seed=42)

    study_svc = optuna.create_study(direction='maximize', sampler=sampler)
    study_svc.optimize(objective_svc, n_trials=n_trials)

    return study_svc.best_params
