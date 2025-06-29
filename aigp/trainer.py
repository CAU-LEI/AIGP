import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from .utils import Timer
from .model_factory import get_model
import os

def save_feature_importance(model, X, output_prefix="feature_importance"):
    """
    Output feature importance after model training:
      - If the model has `feature_importances_` or `coef_`, extract them;
      - Save results as a CSV file and also generate a bar plot (PNG).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    feature_names = X.columns if hasattr(X, "columns") else [f"Feature_{i}" for i in range(X.shape[1])]
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_
        if importance.ndim > 1:
            importance = abs(importance).mean(axis=0)
        else:
            importance = abs(importance)
    else:
        print("This model does not support built-in feature importance.")
        return
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    csv_file = output_prefix + ".csv"
    df_imp.to_csv(csv_file, index=False)
    print("Feature importance saved to", csv_file)
    plt.figure(figsize=(10, 6))
    df_imp.sort_values(by="importance", ascending=False, inplace=True)
    plt.barh(df_imp["feature"], df_imp["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    png_file = output_prefix + ".png"
    plt.savefig(png_file)
    print("Feature importance plot saved to", png_file)
    plt.close()


def pearson_corr(y_true, y_pred):
    """Calculate Pearson correlation coefficient"""
    if len(y_true) < 2:
        return 0
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def get_regression_scorer():
    """Return a Pearson correlation scorer for regression tasks"""
    return make_scorer(pearson_corr, greater_is_better=True)


def run_cross_validation(model, X, y, cv, task_type, n_jobs=1):
    """Evaluate model performance using cross-validation"""
    print("Running {}-fold cross validation...".format(cv))
    start = time.time()
    scorer = get_regression_scorer() if task_type == "regression" else "accuracy"
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)
    elapsed = time.time() - start
    print("Cross validation time: {:.2f} sec".format(elapsed))
    print("Fold scores:", scores)
    print("Average score:", np.mean(scores))
    return scores, np.mean(scores)


def run_train_test(model, X, y, train_size=None, ntest=None, task_type="regression"):
    """Split data into train/test sets, train model, and evaluate"""
    if train_size is None and ntest is None:
        raise ValueError("Must specify train_size or ntest!")

    if train_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    else:
        X_train = X.iloc[:ntest, :]
        y_train = y.iloc[:ntest]
        X_test = X.iloc[ntest:, :]
        y_test = y.iloc[ntest:]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    with Timer("Model training"):
        model.fit(X_train, y_train)
    with Timer("Model prediction"):
        y_pred = model.predict(X_test)

    if task_type == "regression":
        score = pearson_corr(y_test, y_pred)
        print("Pearson correlation:", score)
    else:
        score = accuracy_score(y_test, y_pred)
        print("Accuracy:", score)
    return model, score


def run_grid_search(model, X, y, grid_params, cv, task_type, n_jobs=1):
    """Perform grid search for hyperparameter tuning"""
    print("Starting grid search...")
    scorer = get_regression_scorer() if task_type == "regression" else "accuracy"
    gs = GridSearchCV(estimator=model, param_grid=grid_params, cv=cv, scoring=scorer, n_jobs=n_jobs)
    with Timer("Grid search"):
        gs.fit(X, y)
    print("Best parameters:", gs.best_params_)
    print("Best score:", gs.best_score_)
    return gs.best_estimator_, gs.best_score_, gs.best_params_


# SSA-related functions are already well documented in English. If you'd like me to update their docstrings as well or refactor, let me know.

def train_model(model, X, y, task_type, cv=None, train_size=None, ntest=None,
                grid=False, grid_params=None, ssa=False, ssa_params=None, n_jobs=1, save_checkpoint=""):
    """
    Unified training function supporting cross-validation, grid search, SSA search.
    Splits training data based on either train_size or ntest.
    Saves checkpoint to specified path if save_checkpoint is non-empty.
    """
    extra_info = {}
    if grid and grid_params is not None:
        model, best_score, best_params = run_grid_search(model, X, y, grid_params, cv=cv or 3, task_type=task_type,
                                                         n_jobs=n_jobs)
        extra_info["best_params"] = best_params
        extra_info["best_score"] = best_score
    elif ssa and ssa_params is not None:
        model, best_score, best_params = run_ssa_search(model, X, y, ssa_params, cv=cv or 3, task_type=task_type,
                                                        n_jobs=n_jobs)
        extra_info["best_params"] = best_params
        extra_info["best_score"] = best_score
    if cv:
        scores, avg_score = run_cross_validation(model, X, y, cv, task_type, n_jobs=n_jobs)
        extra_info["cv_scores"] = scores
        extra_info["cv_avg_score"] = avg_score
        with Timer("Training on full data"):
            model.fit(X, y)
    else:
        model, score = run_train_test(model, X, y, train_size, ntest, task_type)

    if save_checkpoint:
        print("Preparing to save model checkpoint to:", save_checkpoint)
        dir_name = os.path.dirname(save_checkpoint)
        if dir_name and not os.path.exists(dir_name):
            print("Directory does not exist. Creating:", dir_name)
            os.makedirs(dir_name)
        try:
            import joblib
            joblib.dump(model, save_checkpoint)
            print("Model checkpoint saved to:", save_checkpoint)
        except Exception as e:
            print("Error saving model:", e)

    return model, score, extra_info
