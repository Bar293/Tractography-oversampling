import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple


# -------------------- DATA LOADING --------------------
def load_data(method: str, data_path: str = "p") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset and labels for a given augmentation method.
    Assumes data is stored as <method>_train_database.npy and <method>_train_labels.npy.
    
    Args:
        method: Name of the augmentation method (e.g., 'smote', 'vae', 'copy').
        data_path: Path where the .npy files are stored.
    
    Returns:
        X: Feature data as numpy array.
        y: Labels as integers (converted from one-hot encoding).
    """
    X = np.load(f"{data_path}/{method}_train_database.npy")
    y_onehot = np.load(f"{data_path}/{method}_train_labels.npy")
    y = np.argmax(y_onehot, axis=1)  # Convert one-hot to class indices
    return X, y


# -------------------- FLATTENING --------------------
def flatten_data(X: np.ndarray) -> np.ndarray:
    """
    Flatten images or high-dimensional data into 2D arrays (samples × features).
    """
    return X.reshape((X.shape[0], -1))


# -------------------- SAMPLING --------------------
def reduce_data(X: np.ndarray, y: np.ndarray, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample a subset of the dataset for visualization.
    
    Args:
        X: Input features.
        y: Labels.
        n_samples: Maximum number of samples to keep.
    
    Returns:
        Subset of X and y with at most n_samples.
    """
    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    return X[idx], y[idx]


# -------------------- DIMENSIONALITY REDUCTION --------------------
def apply_projection(X: np.ndarray, method: str = 'tsne') -> np.ndarray:
    """
    Apply dimensionality reduction (t-SNE or PCA) to reduce data to 2D.
    
    Args:
        X: Input features (2D: samples x features).
        method: 'tsne' or 'pca'.
    
    Returns:
        Projected data in 2D.
    """
    if method == 'tsne':
        print("Computing t-SNE...")
        return TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(X)
    elif method == 'pca':
        return PCA(n_components=2).fit_transform(X)
    else:
        raise ValueError("Invalid method. Use 'tsne' or 'pca'.")


# -------------------- COMPARISON FUNCTION --------------------
def compare_augmentation_methods(
    methods: List[str] = ['smote', 'vae', 'copy'],
    projection: str = 'tsne',
    classes_to_show: List[int] = [13, 14],
    samples_per_method: int = 2000
):
    """
    Compare multiple augmentation methods by projecting data to 2D
    and visualizing selected classes.
    
    Args:
        methods: List of augmentation methods to compare.
        projection: 'tsne' or 'pca' for dimensionality reduction.
        classes_to_show: List of class indices to visualize.
        samples_per_method: Number of samples to project per method.
    """
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes_to_show)))
    all_x, all_y = [], []

    # Collect all projections first to compute global axis limits
    projections = []
    for idx, method in enumerate(methods):
        print(f"\nProcessing method: {method.upper()}")
        X, y = load_data(method)
        X_flat = flatten_data(X)
        X_flat, y = reduce_data(X_flat, y, n_samples=samples_per_method)
        X_proj = apply_projection(X_flat, projection)
        projections.append((X_proj, y))
        all_x.append(X_proj[:, 0])
        all_y.append(X_proj[:, 1])

    # Global limits for consistent axis scaling across plots
    x_min = min([x.min() for x in all_x])
    x_max = max([x.max() for x in all_x])
    y_min = min([y.min() for y in all_y])
    y_max = max([y.max() for y in all_y])

    # Class label mapping for visualization
    label_map = {13: 'FX_L', 14: 'FX_R'}
    for idx, (X_proj, y) in enumerate(projections):
        ax = axes[0][idx]
        method_name = methods[idx].upper()
        if method_name == 'COPY':
            method_name = 'DUPLICATE'
        for i, cls in enumerate(classes_to_show):
            mask = y == cls
            label = label_map.get(cls, f'Class {cls}')
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1], label=label,
                       color=colors[i], s=15, alpha=0.7)
        ax.set_title(f"{method_name} - {projection.upper()}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True)
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"results/map.png", dpi=300)
    plt.close()


# -------------------- EXECUTION --------------------
compare_augmentation_methods(
    methods=['smote', 'vae', 'copy'],
    projection='tsne',
    classes_to_show=[13, 14],
    samples_per_method=20000
)
