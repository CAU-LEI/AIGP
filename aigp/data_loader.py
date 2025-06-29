# aigp/data_loader.py
import os
import subprocess
import pandas as pd


def load_genotype_data(geno_path, geno_sep):
    """
    Automatically detect genotype file format by suffix and load:
      - If it's .txt, read directly;
      - If it's .ped, use plink to convert, then remove the first 6 columns;
      - If it's .vcf, convert to .ped first, then follow the same steps.
    """
    ext = os.path.splitext(geno_path)[1].lower()
    if ext == ".txt":
        X = pd.read_csv(geno_path, sep=geno_sep, header=0)
    elif ext == ".ped":
        base = geno_path[:-4]
        cmd = ["plink", "--file", base, "--recode", "A", "--out", base + "_recode"]
        subprocess.run(cmd, check=True)
        recoded_file = base + "_recode.raw"
        # PLINK raw files are usually space-separated
        X = pd.read_csv(recoded_file, delim_whitespace=True, header=0)
        X = X.iloc[:, 6:]  # Remove the first 6 columns
    elif ext == ".vcf":
        base = geno_path[:-4]
        cmd = ["plink", "--vcf", geno_path, "--recode", "ped", "--out", base + "_vcf"]
        subprocess.run(cmd, check=True)
        ped_file = base + "_vcf.ped"
        base2 = ped_file[:-4]
        cmd = ["plink", "--file", base2, "--recode", "A", "--out", base2 + "_recode"]
        subprocess.run(cmd, check=True)
        recoded_file = base2 + "_recode.raw"
        X = pd.read_csv(recoded_file, delim_whitespace=True, header=0)
        X = X.iloc[:, 6:]
    else:
        raise ValueError("Unknown genotype file format: {}".format(ext))
    return X


def auto_detect_phe_col(phe_df):
    """
    Automatically detect phenotype label column:
      - Prefer columns with headers containing "phenotype" or "trait" (case-insensitive);
      - Otherwise, return the first numeric column;
      - If none found, return column index 0.
    """
    for i, col in enumerate(phe_df.columns):
        if "phenotype" in col.lower() or "trait" in col.lower():
            return i
    for i, col in enumerate(phe_df.columns):
        try:
            pd.to_numeric(phe_df[col])
            return i
        except:
            continue
    return 0


def load_training_data(geno_path, geno_sep, phe_path, phe_sep, phe_col_num, category_cols=None, task_type="regression"):
    """
    Load training data in the following steps:
      1. Load genotype data based on file format.
      2. Load phenotype data and auto-detect label column (if not specified).
      3. Clean phenotype data:
         - For regression: remove samples with y == 0, NA, or empty (also drop corresponding X rows);
         - For classification: remove samples with NA or empty y values.
      4. Extract covariates (fixed effects).
    """
    # Load genotype data
    X = load_genotype_data(geno_path, geno_sep)

    # Load phenotype data
    if phe_path is not None:
        phe_df = pd.read_csv(phe_path, sep=phe_sep, header=0)
    else:
        phe_df = None

    # Auto-detect phenotype column if not provided
    if phe_df is not None and phe_col_num is None:
        phe_col_num = auto_detect_phe_col(phe_df)

    if phe_df is not None and phe_col_num is not None:
        y = phe_df.iloc[:, phe_col_num]
    else:
        y = None

    # Clean phenotype data based on task type
    if y is not None:
        if task_type == "regression":
            valid_idx = y.notna() & (y != "") & (y != 0)
        else:  # classification allows 0
            valid_idx = y.notna() & (y != "")
        y = y[valid_idx]
        if isinstance(X, pd.DataFrame):
            X = X.loc[valid_idx].reset_index(drop=True)
        y = y.reset_index(drop=True)

    # Extract covariates (categorical variables)
    covariates = None
    if phe_df is not None and category_cols is not None:
        covaria
