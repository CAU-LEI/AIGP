# aigp/data_loader.py
import os
import subprocess
import pandas as pd


def load_genotype_data(geno_path, geno_sep):
    """
    根据基因型文件后缀自动检测格式并读取：
      - 若为 txt 格式，直接读取；
      - 若为 .ped 格式，使用 plink 命令进行转换，再去除前 6 列；
      - 若为 .vcf 格式，先转换为 ped 文件，再进行上述处理。
    """
    ext = os.path.splitext(geno_path)[1].lower()
    if ext == ".txt":
        X = pd.read_csv(geno_path, sep=geno_sep, header=0)
    elif ext == ".ped":
        base = geno_path[:-4]
        cmd = ["plink", "--file", base, "--recode", "A", "--out", base + "_recode"]
        subprocess.run(cmd, check=True)
        recoded_file = base + "_recode.raw"
        # plink raw 文件通常以空白分隔
        X = pd.read_csv(recoded_file, delim_whitespace=True, header=0)
        X = X.iloc[:, 6:]  # 去除前6列
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
        raise ValueError("未知的基因型文件格式: {}".format(ext))
    return X


def auto_detect_phe_col(phe_df):
    """
    自动检测表型标签列：
      - 优先查找标题中包含 "phenotype" 或 "trait" 的列（不区分大小写）；
      - 若无则返回第一个数值型的列；
      - 否则返回第 0 列。
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
    读取训练数据，包含以下步骤：
      1. 根据文件格式加载基因型数据。
      2. 读取表型数据，自动检测标签列（若未指定）。
      3. 清洗表型数据：
           - 回归任务：删除 y 值为 0、NA 或空的样本（并删除对应 X 行）。
           - 分类任务：删除 y 值为 NA 或空的样本。
      4. 提取协变量（固定效应）。
    """
    # 读取基因型数据
    X = load_genotype_data(geno_path, geno_sep)

    # 读取表型数据
    if phe_path is not None:
        phe_df = pd.read_csv(phe_path, sep=phe_sep, header=0)
    else:
        phe_df = None

    # 自动检测表型标签列（若未指定）
    if phe_df is not None and phe_col_num is None:
        phe_col_num = auto_detect_phe_col(phe_df)

    if phe_df is not None and phe_col_num is not None:
        y = phe_df.iloc[:, phe_col_num]
    else:
        y = None

    # 根据任务类型清洗表型数据
    if y is not None:
        if task_type == "regression":
            valid_idx = y.notna() & (y != "") & (y != 0)
        else:  # 分类任务允许 0
            valid_idx = y.notna() & (y != "")
        y = y[valid_idx]
        if isinstance(X, pd.DataFrame):
            X = X.loc[valid_idx].reset_index(drop=True)
        y = y.reset_index(drop=True)

    # 提取协变量（分类变量）
    covariates = None
    if phe_df is not None and category_cols is not None:
        covariates = phe_df.iloc[:, category_cols]

    return X, y, covariates


def load_candidate_data(geno_path, geno_sep, phe_path=None, phe_sep=None, category_cols=None):
    """
    读取候选群体数据
    """
    X = load_genotype_data(geno_path, geno_sep)
    covariates = None
    if phe_path is not None and category_cols is not None:
        phe_df = pd.read_csv(phe_path, sep=phe_sep, header=0)
        covariates = phe_df.iloc[:, category_cols]
    return X, covariates


def calculate_geno_stats(X):
    """
    计算基因型数据的统计信息：对每个标记计算缺失率、等位基因频率和最小等位基因频率（MAF）
    返回一个 DataFrame，并可保存为 CSV。
    """
    stats = []
    for col in X.columns:
        series = X[col]
        missing_rate = series.isna().mean()
        nonmissing = series.dropna()
        if len(nonmissing) == 0:
            freq = None
            maf = None
        else:
            # 假定基因型编码为 0,1,2
            freq = nonmissing.sum() / (2 * len(nonmissing))
            maf = min(freq, 1 - freq)
        stats.append({"marker": col, "missing_rate": missing_rate, "allele_frequency": freq, "MAF": maf})
    return pd.DataFrame(stats)


def calculate_phe_stats(y):
    """
    计算表型数据的均值和标准差，并画出分布图保存到当前目录
    """
    import matplotlib.pyplot as plt
    if y.dtype.kind in 'biufc':  # 数值型数据
        mean_val = y.mean()
        std_val = y.std()
        plt.figure()
        y.hist(bins=30)
        plt.title("Phenotype Distribution")
        plt.xlabel("Phenotype")
        plt.ylabel("Frequency")
        plt.savefig("phe_distribution.png")
        plt.close()
        print("表型数据均值: {:.4f}, 标准差: {:.4f}".format(mean_val, std_val))
        return mean_val, std_val
    else:
        print("表型数据不是数值型，无法计算均值和标准差。")
        return None, None
