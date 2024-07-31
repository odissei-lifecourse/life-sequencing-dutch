"""Define prediction models for evaluation"""

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

max_depth = 10
max_features = "log2"
random_state = 395832455
n_estimators = 50
max_iters = 10000

ridge_alpha = 1.0

svm_kernel = 'rbf'
svm_C = 1.0


linear_regression = {
    "single": LinearRegression(),
    "pair":  LogisticRegression(max_iter=max_iters)
}

decision_tree = {
    "single": DecisionTreeRegressor(
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state
        ),
    "pair": DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state
        )
}

random_forest = {
    "single": RandomForestRegressor(
        random_state=random_state,
        max_depth=max_depth,
        n_estimators=n_estimators
        ),
    "pair": RandomForestClassifier(
        random_state=random_state,
        max_depth=max_depth,
        n_estimators=n_estimators
        )
}


support_vector_machine = {
    "single": SVR(kernel=svm_kernel, C=svm_C) ,
    "pair": SVC(kernel=svm_kernel, C=svm_C)
}


ridge_regression = {
    "single":  Ridge(alpha=ridge_alpha),
    "pair": RidgeClassifier(alpha=ridge_alpha)
}


model_dict = {
    "linear_regression": linear_regression,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "support_vector_machine": support_vector_machine,
    "ridge_regression": ridge_regression
}


