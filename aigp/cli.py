# aigp/cli.py
import argparse
import json


def parse_args():
    """
    Parse command-line arguments and return the result
    """
    parser = argparse.ArgumentParser(description="AIGP: Genomic Phenotype Prediction using Machine Learning")

    # Data-related arguments
    parser.add_argument("--geno", type=str, required=True,
                        help="Path to the input genotype feature data file")
    parser.add_argument("--geno_sep", type=str, default=",",
                        help="Separator for the genotype data file, default is ','")
    parser.add_argument("--phe", type=str, default=None,
                        help="Path to the input phenotype/label data file")
    parser.add_argument("--phe_sep", type=str, default=",",
                        help="Separator for the phenotype data file, default is ','")
    parser.add_argument("--phe_col_num", type=int, default=None,
                        help="Column index (0-based) for the phenotype value; auto-detected if not specified")
    parser.add_argument("--category_cols", type=str, default=None,
                        help="Specify categorical covariate columns, separated by commas, e.g., 1,2")

    # Task and model parameters
    parser.add_argument("--type", type=str, choices=["sort", "regression"], required=True,
                        help="Task type: 'sort' (classification) or 'regression'")
    parser.add_argument("--model", type=str, default=None,
                        help="Name of the model to use")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="Model parameters in JSON string format")

    # Dimensionality reduction parameters
    parser.add_argument("--dim_reduction", type=str, choices=["pca", "phate"], default=None,
                        help="Dimensionality reduction method: pca or phate. Default: no reduction")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Number of dimensions after reduction")

    # Data preprocessing (reserved for future use)
    parser.add_argument("--process_x", action="store_true",
                        help="Whether to preprocess feature data (TBD)")
    parser.add_argument("--process_y", action="store_true",
                        help="Whether to preprocess label data (TBD)")

    # Data split and cross-validation
    parser.add_argument("--cv", type=int, default=None,
                        help="Number of folds for cross-validation")
    parser.add_argument("--train_size", type=float, default=None,
                        help="Training set ratio, e.g., 0.8 for 80% training data")
    parser.add_argument("--ntest", type=int, default=None,
                        help="Size of test set, uses first N samples for training")

    # Hyperparameter search options
    parser.add_argument("--grid", action="store_true",
                        help="Enable grid search for hyperparameter tuning")
    parser.add_argument("--grid_model_params", type=str, default="{}",
                        help="Grid search parameters in JSON string format")
    parser.add_argument("--ssa", action="store_true",
                        help="Enable SSA (Sparrow Search Algorithm) for tuning")
    parser.add_argument("--ssa_model_params", type=str, default="{}",
                        help="SSA parameters in JSON string format, e.g.: "
                             "{\"use_custom_ssa\": true, \"param_bounds\": {\"learning_rate\": [0.01, 0.3], "
                             "\"num_leaves\": [10, 50], \"max_depth\": [3, 10]}, \"pop_size\": 20, \"max_iter\": 30}")

    # SHAP analysis parameters
    parser.add_argument("--shap", action="store_true",
                        help="Compute and visualize SHAP values")
    parser.add_argument("--shap_beeswarm", action="store_true",
                        help="Generate SHAP beeswarm plot (requires --shap)")
    parser.add_argument("--shap_feature_heatmap", action="store_true",
                        help="Generate SHAP heatmap (requires --shap)")
    parser.add_argument("--shap_feature_waterfall", action="store_true",
                        help="Generate SHAP waterfall plot (requires --shap)")
    parser.add_argument("--top_features", type=int, default=None,
                        help="Number of top features to show in SHAP plots (requires --shap)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save SHAP plot images")

    # Candidate prediction
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint for candidate prediction")

    # New: Genotype and phenotype statistics and feature importance output
    parser.add_argument("--geno_cal", action="store_true",
                        help="Compute basic genotype statistics like MAF, missing rate, and save to file")
    parser.add_argument("--phe_cal", action="store_true",
                        help="Compute mean and std of phenotype data, plot and save its distribution")
    parser.add_argument("--importance", action="store_true",
                        help="Output feature importance plot and CSV after training")

    # New: Checkpoint saving
    parser.add_argument("--save_checkpoint", type=str, default="",
                        help="Path to save model checkpoint after training (e.g., checkpoint/model.m)")

    # Parallel and GPU options
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of threads for parallel processing")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training; default is CPU")

    args = parser.parse_args()

    # Parse JSON strings (ensure double quotes and lowercase booleans)
    try:
        args.model_params = json.loads(args.model_params)
    except Exception:
        args.model_params = {}
    try:
        args.grid_model_params = json.loads(args.grid_model_params)
    except Exception:
        args.grid_model_params = {}
    try:
        args.ssa_model_params = json.loads(args.ssa_model_params)
    except Exception:
        args.ssa_model_params = {}
    if args.category_cols:
        try:
            args.category_cols = [int(x.strip()) for x in args.category_cols.split(",")]
        except Exception:
            args.category_cols = None

    return args
