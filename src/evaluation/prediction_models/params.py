
RANDOM_SEED = 395832455

regression = {
    "max_iter": 10_000
}

forest = {
    "random_state": RANDOM_SEED,
    "max_depth": 10,
    "n_estimators": 50
}

svc = {
    "kernel": "rbf",
    "C": 0.001
}

ridge = {
    "alpha": 5.0
}

boost = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "absolute_error"
}
