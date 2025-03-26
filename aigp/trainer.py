# aigp/trainer.py
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
    输出模型训练后的特征重要性：
      - 如果模型具有 feature_importances_ 或 coef_ 属性，则提取特征重要性；
      - 将结果保存为 CSV 文件，并生成条形图保存为 PNG 文件。
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
        print("当前模型不支持内置特征重要性输出。")
        return
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    csv_file = output_prefix + ".csv"
    df_imp.to_csv(csv_file, index=False)
    print("特征重要性已保存到", csv_file)
    plt.figure(figsize=(10, 6))
    df_imp.sort_values(by="importance", ascending=False, inplace=True)
    plt.barh(df_imp["feature"], df_imp["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    png_file = output_prefix + ".png"
    plt.savefig(png_file)
    print("特征重要性图已保存到", png_file)
    plt.close()







def pearson_corr(y_true, y_pred):
    """计算皮尔逊相关系数"""
    if len(y_true) < 2:
        return 0
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def get_regression_scorer():
    """返回皮尔逊相关系数 scorer（回归任务）"""
    return make_scorer(pearson_corr, greater_is_better=True)


def run_cross_validation(model, X, y, cv, task_type, n_jobs=1):
    """使用交叉验证评估模型性能"""
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
    """对数据进行训练/测试划分，并训练模型与评估"""
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
    """使用网格搜索调参"""
    print("Starting grid search...")
    scorer = get_regression_scorer() if task_type == "regression" else "accuracy"
    gs = GridSearchCV(estimator=model, param_grid=grid_params, cv=cv, scoring=scorer, n_jobs=n_jobs)
    with Timer("Grid search"):
        gs.fit(X, y)
    print("Best parameters:", gs.best_params_)
    print("Best score:", gs.best_score_)
    return gs.best_estimator_, gs.best_score_, gs.best_params_


def run_ssa_search(model, X, y, ssa_params, cv, task_type, n_jobs=1):
    """
    使用 SSA (麻雀搜索) 进行超参数调参：
    仅支持 LightGBM 和 CatBoost 模型（回归和分类任务均支持）。
    如果 ssa_params 中设置 "use_custom_ssa": true，则调用定制 SSA 版本。
    """
    supported_models = ["LGBMRegressor", "LGBMClassifier", "CatBoostRegressor", "CatBoostClassifier"]
    if ssa_params.get("use_custom_ssa", False) and model.__class__.__name__ in supported_models:
        if model.__class__.__name__ in ["LGBMRegressor", "LGBMClassifier"]:
            best_params, best_metric = run_ssa_search_lgbm(X, y, ssa_params, n_jobs, task_type)
        elif model.__class__.__name__ in ["CatBoostRegressor", "CatBoostClassifier"]:
            best_params, best_metric = run_ssa_search_catboost(X, y, ssa_params, n_jobs, task_type)
        best_model = get_model(task_type, model.__class__.__name__, model_params=best_params, gpu=False,
                               categorical=False)
        with Timer("Training best model (SSA)"):
            best_model.fit(X, y)
        print("SSA best parameters:", best_params)
        print("SSA best metric:", best_metric)
        return best_model, best_metric, best_params
    else:
        # 使用通用 SSA 实现（伪实现）
        import random
        iterations = ssa_params.get("iterations", 10)
        param_grid = ssa_params.get("param_grid", {})
        candidate_params = []
        for _ in range(iterations):
            candidate = {}
            for key, values in param_grid.items():
                candidate[key] = random.choice(values)
            candidate_params.append(candidate)
        from joblib import Parallel, delayed
        def evaluate_candidate(params):
            m = get_model(task_type, model.__class__.__name__, model_params=params, gpu=False, categorical=False)
            scores, avg_score = run_cross_validation(m, X, y, cv=cv or 3, task_type=task_type, n_jobs=n_jobs)
            return params, avg_score

        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_candidate)(params) for params in candidate_params)
        if task_type == "regression":
            best_score = float('inf')
            best_params = None
            for params, score in results:
                if score < best_score:
                    best_score = score
                    best_params = params
        else:
            best_score = -1e9
            best_params = None
            for params, score in results:
                if score > best_score:
                    best_score = score
                    best_params = params
        best_model = get_model(task_type, model.__class__.__name__, model_params=best_params, gpu=False,
                               categorical=False)
        with Timer("Training best model (Generic SSA)"):
            best_model.fit(X, y)
        print("Generic SSA best parameters:", best_params)
        print("Generic SSA best score:", best_score)
        return best_model, best_score, best_params


def run_ssa_search_lgbm(X, y, ssa_params, n_jobs, task_type):
    """
    针对 LightGBM 的 SSA 搜索实现，支持回归和分类任务。
    回归任务使用 RMSE 作为目标；分类任务使用负准确率作为目标。
    """
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import numpy as np
    if task_type == "regression":
        from sklearn.metrics import mean_squared_error
        def objective_function(params):
            model = lgb.LGBMRegressor(
                learning_rate=params['learning_rate'],
                num_leaves=int(params['num_leaves']),
                max_depth=int(params['max_depth']),
                n_estimators=100
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return rmse
    else:
        from sklearn.metrics import accuracy_score
        def objective_function(params):
            model = lgb.LGBMClassifier(
                learning_rate=params['learning_rate'],
                num_leaves=int(params['num_leaves']),
                max_depth=int(params['max_depth']),
                n_estimators=100
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return -acc  # 目标是最小化

    # 划分训练/测试集（比例固定为 0.8 / 0.2，可根据需要调整）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_bounds = ssa_params.get("param_bounds", {
        'learning_rate': (0.01, 0.3),
        'num_leaves': (10, 50),
        'max_depth': (3, 10)
    })
    pop_size = ssa_params.get("pop_size", 20)
    max_iter = ssa_params.get("max_iter", 30)

    class SparrowSearch:
        def __init__(self, obj_func, param_bounds, pop_size=20, max_iter=30):
            self.obj_func = obj_func
            self.param_bounds = param_bounds
            self.pop_size = pop_size
            self.max_iter = max_iter
            self.dim = len(param_bounds)
            self.lb = np.array([param_bounds[k][0] for k in param_bounds])
            self.ub = np.array([param_bounds[k][1] for k in param_bounds])

        def optimize(self):
            pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
            fitness = np.array([self.obj_func(dict(zip(self.param_bounds.keys(), p))) for p in pop])
            for t in range(self.max_iter):
                best_idx = np.argmin(fitness)
                worst_idx = np.argmax(fitness)
                for i in range(self.pop_size):
                    r1 = np.random.rand()
                    if r1 < 0.8:
                        pop[i] += np.random.randn(self.dim) * (pop[best_idx] - pop[i])
                    else:
                        pop[i] += np.random.randn(self.dim)
                for i in range(self.pop_size):
                    if i == worst_idx:
                        pop[i] += np.random.randn(self.dim) * (pop[best_idx] - pop[i])
                pop = np.clip(pop, self.lb, self.ub)
                fitness = np.array([self.obj_func(dict(zip(self.param_bounds.keys(), p))) for p in pop])
            best_idx = np.argmin(fitness)
            best_params = dict(zip(self.param_bounds.keys(), pop[best_idx]))
            return best_params, np.min(fitness)

    ssa = SparrowSearch(objective_function, param_bounds, pop_size=pop_size, max_iter=max_iter)
    best_params, best_metric = ssa.optimize()
    return best_params, best_metric


def run_ssa_search_catboost(X, y, ssa_params, n_jobs, task_type):
    """
    针对 CatBoost 的 SSA 搜索实现，支持回归和分类任务。
    回归任务使用 RMSE 作为目标；分类任务使用负准确率作为目标。
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    if task_type == "regression":
        from sklearn.metrics import mean_squared_error
        def objective_function(params):
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                learning_rate=params['learning_rate'],
                depth=int(params['depth']),
                iterations=int(params['iterations']),
                verbose=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return rmse
    else:
        from sklearn.metrics import accuracy_score
        def objective_function(params):
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                learning_rate=params['learning_rate'],
                depth=int(params['depth']),
                iterations=int(params['iterations']),
                verbose=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return -acc

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_bounds = ssa_params.get("param_bounds", {
        'learning_rate': (0.01, 0.3),
        'depth': (3, 10),
        'iterations': (100, 500)
    })
    pop_size = ssa_params.get("pop_size", 20)
    max_iter = ssa_params.get("max_iter", 30)

    class SparrowSearch:
        def __init__(self, obj_func, param_bounds, pop_size=20, max_iter=30):
            self.obj_func = obj_func
            self.param_bounds = param_bounds
            self.pop_size = pop_size
            self.max_iter = max_iter
            self.dim = len(param_bounds)
            self.lb = np.array([param_bounds[k][0] for k in param_bounds])
            self.ub = np.array([param_bounds[k][1] for k in param_bounds])

        def optimize(self):
            pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
            fitness = np.array([self.obj_func(dict(zip(self.param_bounds.keys(), p))) for p in pop])
            for t in range(self.max_iter):
                best_idx = np.argmin(fitness)
                worst_idx = np.argmax(fitness)
                for i in range(self.pop_size):
                    r1 = np.random.rand()
                    if r1 < 0.8:
                        pop[i] += np.random.randn(self.dim) * (pop[best_idx] - pop[i])
                    else:
                        pop[i] += np.random.randn(self.dim)
                for i in range(self.pop_size):
                    if i == worst_idx:
                        pop[i] += np.random.randn(self.dim) * (pop[best_idx] - pop[i])
                pop = np.clip(pop, self.lb, self.ub)
                fitness = np.array([self.obj_func(dict(zip(self.param_bounds.keys(), p))) for p in pop])
            best_idx = np.argmin(fitness)
            best_params = dict(zip(self.param_bounds.keys(), pop[best_idx]))
            return best_params, np.min(fitness)

    ssa = SparrowSearch(objective_function, param_bounds, pop_size=pop_size, max_iter=max_iter)
    best_params, best_metric = ssa.optimize()
    return best_params, best_metric


def train_model(model, X, y, task_type, cv=None, train_size=None, ntest=None,
                grid=False, grid_params=None, ssa=False, ssa_params=None, n_jobs=1, save_checkpoint=""):
    """
    综合训练函数，支持交叉验证、网格搜索、SSA 搜索，
    并根据传入的参数进行训练集划分（通过 train_size 或 ntest）。
    如果 save_checkpoint 非空，则在训练结束后保存模型到指定路径。
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
        print("准备保存模型检查点到:", save_checkpoint)
        dir_name = os.path.dirname(save_checkpoint)
        if dir_name and not os.path.exists(dir_name):
            print("目录不存在，创建目录:", dir_name)
            os.makedirs(dir_name)
        try:
            import joblib
            joblib.dump(model, save_checkpoint)
            print("Model checkpoint saved to:", save_checkpoint)
        except Exception as e:
            print("保存模型时出错:", e)

    return model, score, extra_info
