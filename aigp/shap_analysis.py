import shap
import matplotlib.pyplot as plt
import os
from .utils import Timer


def analyze_shap(model, X, feature_names=None, output=None,
                 shap_beeswarm=False, shap_feature_heatmap=False, shap_feature_waterfall=False,
                 top_features=None):
    """
    Compute and visualize SHAP values. Supports beeswarm plot, heatmap, and waterfall plot.

    Parameters:
      model: Trained model
      X: Feature data used for computing SHAP values (preferably raw or reduced)
      feature_names: List of feature names
      output: Folder path to save the images
      shap_beeswarm: Whether to generate a beeswarm plot
      shap_feature_heatmap: Whether to generate a heatmap of features
      shap_feature_waterfall: Whether to generate a waterfall plot (only for a single sample)
      top_features: Number of top important features to display (for some plots)
    """
    # Choose explainer: use TreeExplainer for tree models, otherwise KernelExplainer or default
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model.predict, X)

    with Timer("Computing SHAP values"):
        shap_values = explainer.shap_values(X)

    # Create output directory if it doesn't exist
    if output and not os.path.exists(output):
        os.makedirs(output)

    # Beeswarm plot
    if shap_beeswarm:
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=top_features, show=False)
        if output:
            plt.savefig(os.path.join(output, "shap_beeswarm.png"))
        else:
            plt.show()
        plt.close()

    # Feature heatmap
    if shap_feature_heatmap:
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="heatmap", max_display=top_features, show=False)
        if output:
            plt.savefig(os.path.join(output, "shap_heatmap.png"))
        else:
            plt.show()
        plt.close()

    # Waterfall plot (only works for a single sample)
    if shap_feature_waterfall:
        idx = 0  # Use the first sample
        plt.figure()
        shap.plots.waterfall(shap_values[idx], max_display=top_features, show=False)
        if output:
            plt.savefig(os.path.join(output, "shap_waterfall.png"))
        else:
            plt.show()
        plt.close()

    return shap_values
