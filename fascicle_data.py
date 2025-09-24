import os
import numpy as np
from dipy.io.streamline import load_tractogram
from math import floor
import itertools
import tensorflow as tf

VALID_TRACT_FORMATS = ('.trk', '.tck', '.vtk', '.vtp', '.fib', '.dpy')

# -------------------- LOAD FASCICLE STREAMLINES --------------------
def load_fascicle_streamlines(subject_path, fascicle_name):
    """
    Load all streamlines for a given fascicle in a subject folder.
    """
    streamlines = []
    for ext in VALID_TRACT_FORMATS:
        tract_file = os.path.join(subject_path, f"{fascicle_name}{ext}")
        if os.path.exists(tract_file):
            tractogram = load_tractogram(tract_file, 'same', bbox_valid_check=False)
            streamlines.extend(tractogram.streamlines)
            break
    return streamlines

# -------------------- RESAMPLE STREAMLINES --------------------
def resample_streamline(stream, n_points):
    """
    Resample a streamline to exactly n_points.
    """
    c = max(floor(len(stream) / n_points), 1)
    new_stream = stream[::c]
    if len(new_stream) > n_points:
        new_stream = new_stream[:n_points]
    new_stream[-1] = stream[-1]
    return new_stream.astype('float32')

# -------------------- SIMPLIFY STREAMLINES --------------------
def simplify_streamlines(streamlines):
    """
    Remove duplicate streamlines.
    """
    data_to_list = [tuple(map(tuple, s.tolist())) for s in streamlines]
    data_to_list.sort()
    return [np.array(k) for k, _ in itertools.groupby(data_to_list)]

# -------------------- COLLECT FASCICLE DATA --------------------
def collect_fascicle_data(paths, fascicle_name, th=15, n_points=15):
    """
    Collect all streamlines of one fascicle from multiple dataset paths.

    Args:
        paths (list): List of dataset folders containing subjects.
        fascicle_name (str): Fascicle to extract.
        th (int): Minimum number of points per streamline.
        n_points (int): Resample all streamlines to this number of points.

    Returns:
        dataset: np.array of shape (N, n_points, 3)
        labels: np.array of fascicle labels (all same)
    """
    all_streamlines = []

    for path in paths:
        subjects = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for subject in subjects:
            subject_path = os.path.join(path, subject)
            streamlines = load_fascicle_streamlines(subject_path, fascicle_name)
            # Filter short streamlines and resample
            streamlines = [resample_streamline(s, n_points) for s in streamlines if len(s) >= th]
            all_streamlines.extend(streamlines)

    # Remove duplicates
    all_streamlines = simplify_streamlines(all_streamlines)
    dataset = np.array(all_streamlines, dtype='float32')
    labels = np.array([fascicle_name] * len(all_streamlines))

    print(f"Collected {len(dataset)} streamlines for fascicle {fascicle_name}")
    return dataset, labels

# -------------------- PREPROCESSING --------------------
def preprocess_labels(labels, fascicle_name):
    """
    Convert labels to one-hot encoding.
    """
    labels_onehot = tf.keras.utils.to_categorical(np.zeros(len(labels)), num_classes=1)
    return labels_onehot

# -------------------- PROCESS TRAIN AND TEST --------------------
if __name__ == "__main__":
    # Paths to multiple datasets
    paths = ["Tractoinferno/trainset", "Tractoinferno/testset"]
    fascicle_name = "FX_L"  # Extract only this fascicle

    # Collect data
    dataset, labels = collect_fascicle_data(paths, fascicle_name, th=15, n_points=15)
    labels_onehot = preprocess_labels(labels, fascicle_name)

    # Save to .npy files
    np.save("data/fx_l_dataset.npy", dataset)
    np.save("data/fx_l_labels.npy", labels_onehot)
    print(f"Saved dataset with shape {dataset.shape} and labels {labels_onehot.shape}")

    # Paths to multiple datasets
    paths = ["Tractoinferno/trainset", "Tractoinferno/testset"]
    fascicle_name = "FX_R"  # Extract only this fascicle

    # Collect data
    dataset, labels = collect_fascicle_data(paths, fascicle_name, th=15, n_points=15)
    labels_onehot = preprocess_labels(labels, fascicle_name)
    
    # Save to .npy files
    np.save("data/fx_r_dataset.npy", dataset)
    np.save("data/fx_r_labels.npy", labels_onehot)
    print(f"Saved dataset with shape {dataset.shape} and labels {labels_onehot.shape}")