#!/usr/bin/env python
# main1.py
"""
AIGP Main Entry Point
Usage examples:
  Train a regression model:
    python main1.py --geno data/test_x.txt --geno_sep "\t" --phe data/test_y.txt --phe_sep "\s" --phe_col_num 1 --type regression --model LinearRegression --train_size 0.8
  Train a classification model with PCA dimensionality reduction and grid search:
    python main1.py --geno data/test_x.txt --geno_sep "\t" --phe data/test_y.txt --phe_sep "\s" --phe_col_num 1 --type sort --model LogisticRegression --dim_reduction pca --n_components 20 --grid --grid_model_params "{\"fit_intercept\": [true, false], \"C\": [1.0, 100]}" --n_jobs 4
  (Note: JSON parameters must follow standard format)
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

    # Load training data with automatic phenotype column detection and data cleaning (based on task type)
    X, y, covariates = load_training_data(args.geno, args.geno_sep, args.phe, args.phe_sep,
                                          args.phe_col_num, args.category_cols, task_type=args.type)

    # Compute genotype statistics (if specified)
    if args.geno_cal:
        geno_stats = calculate_geno_stats(X)
        geno_stats.to_csv("geno_stats.csv", index=False)
        print("Genotype statistics saved to geno_stats.csv")

    # Compute phenotype statistics (if specified)
    if args.phe_cal and y is not None:
        calculate_phe_stats(y)

    # Dimensionality reduction if specified
    if args.dim_reduction and args.n_components:
        print("Performing {} dimensionality reduction to {} components".format(args.dim_reduction, args.n_components))
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

    # Create model
    model = get_model(task_type="regression" if args.type == "regression" else "sort",
                      model_name=args.model,
                      model_params=args.model_params,
                      gpu=args.gpu,
                      categorical=(args.category_cols is not None))

    # Train model with support for cross-validation, grid search, SSA search, and parallel computation
    model, score, extra_info = train_model(model, X, y,
                                           task_type="regression" if args.type == "regression" else "sort",
                                           cv=args.cv, train_size=args.train_size, ntest=args.ntest,
                                           grid=args.grid, grid_params=args.grid_model_params,
                                           ssa=args.ssa, ssa_params=args.ssa_model_params,
                                           n_jobs=args.n_jobs, save_checkpoint=args.save_checkpoint)

    # SHAP analysis
    if args.shap:
        print("Starting SHAP analysis...")
        feature_names = X.columns if hasattr(X, "columns") else None
        analyze_shap(model, X, feature_names=feature_names, output=args.output,
                     shap_beeswarm=args.shap_beeswarm,
                     shap_feature_heatmap=args.shap_feature_heatmap,
                     shap_feature_waterfall=args.shap_feature_waterfall,
                     top_features=args.top_features)

    # Output feature importance
    if args.importance:
        save_feature_importance(model, X)


if __name__ == "__main__":
    args = parse_args()
    print("save_checkpoint:", args.save_checkpoint)
    main()


"""
python main1.py --geno data/test_x.txt --geno_sep "\t" --phe data/test_y.txt --phe_sep "\s" --phe_col_num 1 --type sort --model LogisticRegression --dim_reduction pca --n_components 20 --grid --grid_model_params "{\"fit_intercept\": [true, false], \"C\": [1.0, 100]}" --geno_cal --phe_cal --n_jobs 4 --train_size 0.8
"""
