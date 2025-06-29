from sklearn.decomposition import PCA
import phate


def reduce_dimensions(X, method, n_components):
    """
    Perform dimensionality reduction on data X

    Parameters:
      X: Feature data, either a DataFrame or ndarray
      method: Reduction method, either "pca" or "phate"
      n_components: Number of components to reduce to

    Returns:
      X_reduced: Dimensionally reduced data
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method == "phate":
        reducer = phate.PHATE(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("Unsupported dimensionality reduction method: {}".format(method))
    return X_reduced
