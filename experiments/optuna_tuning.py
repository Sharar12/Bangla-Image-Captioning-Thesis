import optuna
import torch

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    # Placeholder objective evaluation
    score = (1.0 - dropout) * 30.0 + (hidden_dim / 768.0) * 10.0
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best params:", study.best_params)
