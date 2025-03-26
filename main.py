
#!/usr/bin/env python
# main1.py
"""
AIGP 主程序入口
使用示例：
  训练回归模型：
    python main1.py --geno data/test_x.txt --geno_sep "\t" --phe data/test_y.txt --phe_sep "\s" --phe_col_num 1 --type regression --model LinearRegression --train_size 0.8
  训练分类模型，使用 PCA 降维，并进行网格搜索：
    python main1.py --geno data/test_x.txt --geno_sep "\t" --phe data/test_y.txt --phe_sep "\s" --phe_col_num 1 --type sort --model LogisticRegression --dim_reduction pca --n_components 20 --grid --grid_model_params "{\"fit_intercept\": [true, false], \"C\": [1.0, 100]}" --n_jobs 4
  （注意 JSON 参数必须符合标准格式）
"""


import matplotlib
matplotlib.use('Agg')
from aigp.cli import parse_args
from aigp.data_loader import load_training_data, calculate_geno_stats, calculate_phe_stats
from aigp.dim_reduction import reduce_dimensions
from aigp.model_factory import get_model
from aigp.trainer import train_model, save_feature_importance
from aigp.shap_analysis import analyze_shap


def main():
    args = parse_args()

    # 读取训练数据，同时进行表型列自动检测和数据清洗（根据任务类型）
    X, y, covariates = load_training_data(args.geno, args.geno_sep, args.phe, args.phe_sep,
                                          args.phe_col_num, args.category_cols, task_type=args.type)

    # 计算基因型数据统计信息（若指定）
    if args.geno_cal:
        geno_stats = calculate_geno_stats(X)
        geno_stats.to_csv("geno_stats.csv", index=False)
        print("基因型统计信息已保存到 geno_stats.csv")

    # 计算表型统计信息（若指定）
    if args.phe_cal and y is not None:
        calculate_phe_stats(y)

    # 降维：若指定了降维方法和目标维度
    if args.dim_reduction and args.n_components:
        print("进行 {} 降维，目标维度：{}".format(args.dim_reduction, args.n_components))
        X_reduced = reduce_dimensions(X, args.dim_reduction, args.n_components)
        import pandas as pd
        X_reduced = pd.DataFrame(X_reduced, columns=["PC{}".format(i + 1) for i in range(args.n_components)])
        if covariates is not None:
            X = pd.concat([X_reduced, covariates.reset_index(drop=True)], axis=1)
        else:
            X = X_reduced
    else:
        if covariates is not None:
            import pandas as pd
            X = pd.concat([X.reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)

    # 构造模型
    model = get_model(task_type="regression" if args.type == "regression" else "sort",
                      model_name=args.model,
                      model_params=args.model_params,
                      gpu=args.gpu,
                      categorical=(args.category_cols is not None))

    # 模型训练：支持交叉验证、网格搜索、SSA 搜索，传入 n_jobs 参数实现并行计算
    model, score, extra_info = train_model(model, X, y,
                                           task_type="regression" if args.type == "regression" else "sort",
                                           cv=args.cv, train_size=args.train_size, ntest=args.ntest,
                                           grid=args.grid, grid_params=args.grid_model_params,
                                           ssa=args.ssa, ssa_params=args.ssa_model_params,
                                           n_jobs=args.n_jobs,save_checkpoint=args.save_checkpoint)

    # SHAP 分析
    if args.shap:
        print("开始 SHAP 分析...")
        feature_names = X.columns if hasattr(X, "columns") else None
        analyze_shap(model, X, feature_names=feature_names, output=args.output,
                     shap_beeswarm=args.shap_beeswarm,
                     shap_feature_heatmap=args.shap_feature_heatmap,
                     shap_feature_waterfall=args.shap_feature_waterfall,
                     top_features=args.top_features)

    # 特征重要性输出
    if args.importance:
        save_feature_importance(model, X)


if __name__ == "__main__":
    args = parse_args()
    print("save_checkpoint:", args.save_checkpoint)
    main()


"""
python main1.py --geno data/test_x.txt --geno_sep "\t" --phe data/test_y.txt --phe_sep "\s" --phe_col_num 1 --type sort --model LogisticRegression --dim_reduction pca --n_components 20 --grid --grid_model_params "{\"fit_intercept\": [true, false], \"C\": [1.0, 100]}" --geno_cal --phe_cal --n_jobs 4 --train_size 0.8


"""