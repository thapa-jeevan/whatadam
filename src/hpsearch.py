import mlflow
import optuna

from train_run import run

experiment = "cifar"
approach = "hat_adamw"

def objective(trial: optuna.Trial) -> float:
    nepochs = 200
    hp = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "lr_min": trial.suggest_float("lr_min", 1e-6, 1e-3, log=True),
        "lr_factor": trial.suggest_categorical("lr_factor", [1.5, 2, 2.5, 3, 5, 7.5, 10.0]),
        "lr_patience": trial.suggest_int("lr_patience", 2, 10),
        "clipgrad": trial.suggest_int("clipgrad", 1, 20_000, log=True),
        "lamb": trial.suggest_float("lamb", 0.001, 10),
        "smax": trial.suggest_categorical("smax", [25, 50, 100, 200, 400, 800]),
        "sbatch": trial.suggest_categorical("sbatch", [64, 128, 256]),
    }
    acc_ls = run(
        experiment, approach, nepochs, hp["lr"], hp["lamb"], hp["sbatch"], hp["lr_min"], hp["lr_factor"],
        hp["lr_patience"], hp["clipgrad"], hp["smax"]
    )
    final_mean_acc = acc_ls[-1].mean()

    with mlflow.start_run():
        mlflow.set_tag("approach", approach)
        mlflow.set_tag("experiment", experiment)
        mlflow.log_param("nepochs", nepochs)
        mlflow.log_params(hp)

        for key, value in hp.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("final_mean_acc", final_mean_acc)
        for i, acc in enumerate(acc_ls[-1]):
            mlflow.log_metric(f"acc_task_{i}", acc)

    return final_mean_acc


EXPERIMENT_NAME = f"HAT-hyperparam-search-{approach}-{experiment}"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

study = optuna.create_study(
    study_name="hat_search",
    direction="maximize",
)

study.optimize(
    objective,
    n_trials=300,
    n_jobs=1
)

print("Best score :", 1 - study.best_value)
print("Best params:", study.best_params)
