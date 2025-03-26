# aigp/shap_analysis.py
import shap
import matplotlib.pyplot as plt
import os
from .utils import Timer


def analyze_shap(model, X, feature_names=None, output=None,
                 shap_beeswarm=False, shap_feature_heatmap=False, shap_feature_waterfall=False,
                 top_features=None):
    """
    计算并可视化 SHAP 值。支持蜂群图、热图、瀑布图。

    参数：
      model: 已训练好的模型
      X: 用于计算 SHAP 值的特征数据（最好为原始数据或降维后数据）
      feature_names: 特征名称列表
      output: 保存图片的文件夹路径
      shap_beeswarm: 是否生成蜂群图
      shap_feature_heatmap: 是否生成特征热图
      shap_feature_waterfall: 是否生成瀑布图（仅针对单一样本）
      top_features: 显示前 N 个重要特征（用于部分图形）
    """
    # 选择解释器，若为树模型则使用 TreeExplainer，否则使用 KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model.predict, X)

    with Timer("计算 SHAP 值"):
        shap_values = explainer.shap_values(X)

    # 若指定输出路径但不存在则创建
    if output and not os.path.exists(output):
        os.makedirs(output)

    # 蜂群图
    if shap_beeswarm:
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=top_features, show=False)
        if output:
            plt.savefig(os.path.join(output, "shap_beeswarm.png"))
        else:
            plt.show()
        plt.close()

    # 特征热图
    if shap_feature_heatmap:
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="heatmap", max_display=top_features, show=False)
        if output:
            plt.savefig(os.path.join(output, "shap_heatmap.png"))
        else:
            plt.show()
        plt.close()

    # 瀑布图（仅对单一样本有效）
    if shap_feature_waterfall:
        # 选择第一个样本
        idx = 0
        plt.figure()
        shap.plots.waterfall(shap_values[idx], max_display=top_features, show=False)
        if output:
            plt.savefig(os.path.join(output, "shap_waterfall.png"))
        else:
            plt.show()
        plt.close()

    return shap_values
