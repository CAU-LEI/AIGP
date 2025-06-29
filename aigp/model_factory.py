from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesClassifier
import warnings

# Try importing xgboost
try:
    from xgboost import XGBRegressor, XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


def get_model(task_type, model_name, model_params=None, gpu=False, categorical=False):
    """
    Return a model instance based on task type and model name

    Parameters:
      task_type: "regression" or "sort"
      model_name: Model name string (e.g., "LinearRegression", "LogisticRegression", "CatBoostClassifier", etc.)
      model_params: Dictionary of model parameters
      gpu: Whether to use GPU for training
      categorical: Whether there are categorical covariates (if present, only LightGBM or CatBoost is allowed)

    Returns:
      model: Initialized model instance
    """
    model_params = model_params or {}

    # If categorical variables are present, only CatBoost or LightGBM models are allowed
    if categorical:
        allowed = ["CatBoost", "CatBoostClassifier", "CatBoostRegressor", "LGBM", "LGBMClassifier", "LGBMRegressor"]
        if model_name not in allowed:
            raise ValueError("When categorical variables are present, only LightGBM or CatBoost models are supported!")

    # Build model dictionary based on task type
    if task_type == "regression":
        models = {
            "knn": lambda: KNeighborsRegressor(**model_params),
            "svm": lambda: SVR(**model_params),
            "LinearRegression": lambda: LinearRegression(**model_params),
            "RidgeRegression": lambda: Ridge(**model_params),
            "ElasticNet": lambda: ElasticNet(**model_params),
            "RandomForest": lambda: RandomForestRegressor(**model_params),
            "GradientBoosting": lambda: GradientBoostingRegressor(**model_params),
            "AdaBoost": lambda: AdaBoostRegressor(**model_params),
            "CatBoostRegressor": lambda: CatBoostRegressor(**(add_gpu_params(model_params, gpu, model_type="catboost")),
                                                           verbose=0),
            "LGBMRegressor": lambda: LGBMRegressor(**(add_gpu_params(model_params, gpu, model_type="lgbm")))
        }
        if xgboost_available:
            models["xgboost"] = lambda: XGBRegressor(**(add_gpu_params(model_params, gpu, model_type="xgboost")))
        else:
            warnings.warn("xgboost package not installed. xgboost model will be unavailable.")
    elif task_type == "sort":  # classification
        models = {
            "knn": lambda: KNeighborsClassifier(**model_params),
            "svm": lambda: SVC(**model_params),
            "LogisticRegression": lambda: LogisticRegression(**model_params, max_iter=1000),
            "RandomForest": lambda: RandomForestClassifier(**model_params),
            "GradientBoosting": lambda: GradientBoostingClassifier(**model_params),
            "AdaBoost": lambda: AdaBoostClassifier(**model_params),
            "CatBoost": lambda: CatBoostClassifier(**(add_gpu_params(model_params, gpu, model_type="catboost")),
                                                   verbose=0),
            "LGBM": lambda: LGBMClassifier(**(add_gpu_params(model_params, gpu, model_type="lgbm"))),
            "ExtraTrees": lambda: ExtraTreesClassifier(**model_params)
        }
        if xgboost_available:
            models["xgboost"] = lambda: XGBClassifier(**(add_gpu_params(model_params, gpu, model_type="xgboost")))
        else:
            warnings.warn("xgboost package not installed. xgboost model will be unavailable.")
    else:
        raise ValueError("Unknown task type: {}".format(task_type))

    if model_name not in models:
        raise ValueError("Unsupported model name: {}. Available models: {}".format(model_name, list(models.keys())))

    return models[model_name]()


def add_gpu_params(params, gpu, model_type):
    """
    Update model parameters based on the gpu flag

    Parameters:
      params: Original parameter dictionary
      gpu: Whether to use GPU
      model_type: Type of model - "catboost", "lgbm", or "xgboost"

    Returns:
      Updated parameter dictionary
    """
    params = params.copy()
    if gpu:
        if model_type == "catboost":
            params["task_type"] = "GPU"
        elif model_type == "lgbm":
            params["device"] = "gpu"
        elif model_type == "xgboost":
            params["tree_method"] = "gpu_hist"
    return params
