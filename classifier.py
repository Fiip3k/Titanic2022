from xgboost import XGBClassifier
from sklearn import model_selection


def get_model():
    model = XGBClassifier(objective="binary:logistic")
    return model


def fit_model(X_train, y_train):
    model = get_model().fit(X_train, y_train)
    return model


def fit_grid_search(X_train, y_train):
    param_grid = {
        "max_depth": [3, 4, 5, 7],
        "learning_rate": [0.1, 0.01, 0.05],
        "gamma": [0, 0.25, 1],
        "reg_lambda": [0, 1, 10],
        "scale_pos_weight": [1, 3, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.5],
    }

    grid = model_selection.GridSearchCV(
        get_model(), param_grid, n_jobs=-1, cv=3, scoring="roc_auc")

    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    return model


def get_predictions(model, X_test):
    y_test = model.predict(X_test)
    return y_test
