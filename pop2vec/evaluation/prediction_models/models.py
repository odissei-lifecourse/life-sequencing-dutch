"""Define prediction models for evaluation"""

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from pop2vec.evaluation.prediction_models import params # assumes code is run from `LifeCourseEvaluation/`

linear_regression = {
    "single": LinearRegression(),
    "pair":  LogisticRegression(**params.regression)
}

random_forest = {
    "single": RandomForestRegressor(**params.forest),
    "pair": RandomForestClassifier(**params.forest)
}

support_vector_machine = {
    "single": SVR(**params.svc) ,
    "pair": SVC(**params.svc)
}


ridge_regression = {
    "single":  Ridge(**params.ridge),
    "pair": RidgeClassifier(**params.ridge)
}

gradient_boost = {
    "single": GradientBoostingRegressor(**params.boost),
    "pair": GradientBoostingClassifier(**params.boost)
}


model_dict = {
    "linear_regression": linear_regression,
    "random_forest": random_forest,
    "support_vector_machine": support_vector_machine,
    "ridge_regression": ridge_regression,
    "gradient_boost": gradient_boost
}


