import itertools
from dipy.io.streamline import load_tractogram
import numpy as np
from math import floor
import os
import random
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler

# Valid tractography file formats
VALID_TRACT_FORMATS = ('.trk', '.tck', '.vtk', '.vtp', '.fib', '.dpy')

# -------------------- FUNCTIONS --------------------
def load_fascicle_tractogram(subject, subject_path, tract_name, e_nii=None):
    """
    Load the tractogram for a given fascicle and subject.
    Supports multiple file formats (.trk, .tck, etc.) and optional reference NIfTI.
    """
    tract_file = None
    for ext in VALID_TRACT_FORMATS:
        possible_file = os.path.join(subject_path, f"{tract_name}{ext}")
        if os.path.exists(possible_file):
            tract_file = possible_file
            break
    if not tract_file:
        return []

    if tract_file.endswith('.trk') and e_nii is None:
        tractogram = load_tractogram(tract_file, 'same', bbox_valid_check=False)
    elif tract_file.endswith('.trk') and e_nii is not None:
        tractogram = load_tractogram(tract_file, 'same', bbox_valid_check=False, trk_header_check=True)
    else:
        if e_nii is None:
            return []
        nii_path = os.path.join(subject_path, "anat", e_nii)
        if not os.path.exists(nii_path):
            return []
        tractogram = load_tractogram(tract_file, nii_path, bbox_valid_check=False)

    return tractogram.streamlines

def p_even(n_points, stream):
    """
    Resample a streamline evenly to n_points.
    Ensures the last point is preserved.
    """
    c = max(floor(len(stream) / n_points), 1)
    new_stream = stream[::c]
    if len(new_stream) > n_points:
        new_stream = new_stream[:n_points]
    new_stream[-1] = stream[-1]
    return new_stream.astype('float32')

def simplify_database(data):
    """
    Remove duplicate streamlines from a list of streamlines.
    """
    # Convert each streamline to tuple of tuples for hashing
    data_to_list = [tuple(map(tuple, stream.tolist())) for stream in data]
    data_to_list.sort()
    return [np.array(k) for k, _ in itertools.groupby(data_to_list)]

def preprocessing(dataset, labels):
    """
    Convert dataset to float32 and labels to one-hot encoding.
    """
    dataset = dataset.astype('float32')
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(bundles_names))
    dataset, labels = np.array(dataset), np.array(labels)
    print(f"Dataset shape: {dataset.shape}, Labels shape: {labels.shape}")
    return dataset, labels

def get_subjects(path):
    """
    Return list of subject folders in a dataset path.
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def undersample_batches_subjects(bundles_names, base_path, subjects, th, p, batch_size=2, random_state=42):
    """
    Load tractograms in batches per subject, resample and undersample using RandomUnderSampler.
    Returns final dataset and labels.
    """
    dataset, labels = [], []
    num_subjects = len(subjects)

    for batch_start in range(0, num_subjects, batch_size):
        print(f"Processing batch {batch_start//batch_size + 1}/{(num_subjects + batch_size - 1)//batch_size}")
        batch_subjects = subjects[batch_start:batch_start + batch_size]

        batch_data, batch_labels = [], []

        for u, bundle_name in bundles_names.items():
            fascicle = []
            for subject in batch_subjects:
                subject_path = os.path.join(base_path, subject)
                ST = load_fascicle_tractogram(subject, subject_path, bundle_name)
                # Resample each streamline and filter by length threshold
                ST_p = [p_even(p, stream) for stream in ST if len(stream) >= th]
                fascicle += ST_p

            # Remove duplicate streamlines
            fascicle_simplified = simplify_database(np.array(fascicle))
            batch_data += fascicle_simplified
            batch_labels += [u] * len(fascicle_simplified)

        if len(batch_data) == 0:
            continue

        # Flatten streamlines for undersampling
        X_flat = np.array([s.flatten() for s in batch_data])
        y_arr = np.array(batch_labels)

        # Apply random undersampling
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X_flat, y_arr)

        n_points = batch_data[0].shape[0]
        X_res_3d = X_res.reshape((-1, n_points, 3))

        dataset.extend(X_res_3d)
        labels.extend(y_res)

        print(f"Batch {batch_start//batch_size + 1}: {len(X_res_3d)} samples after undersampling")

    dataset, labels = np.array(dataset), np.array(labels)
    print(f"Total final dataset: {dataset.shape}, Total labels: {labels.shape}")

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Final samples per class: {dict(zip(unique, counts))}")

    # Compute mean number of samples per bundle excluding bundles 13 and 14
    exclude = [13, 14]
    filtered_counts = [c for u, c in zip(unique, counts) if u not in exclude]
    if filtered_counts:
        mean_count = np.mean(filtered_counts)
        print(f"Mean number of samples per bundle (excluding 13 and 14): {mean_count:.2f}")
    else:
        print("No bundles to compute mean (excluding 13 and 14)")

    return dataset, labels

# -------------------- PARAMETERS --------------------
trainpath = "Tractoinferno/trainset"
testpath = "Tractoinferno/testset"
p, th = 15, 20  # Resample points and minimum streamline length

# -------------------- BUNDLES NAMES --------------------
bundles_names = {
    0: 'AF_L', 1: 'AF_R', 2: 'CC_Oc', 3: 'CC_Fr_1', 4: 'CC_Fr_2',
    5: 'CC_Pa', 6: 'CC_Pr_Po', 7: 'CG_L', 8: 'CG_R', 9: 'FAT_L',
    10: 'FAT_R', 11: 'FPT_L', 12: 'FPT_R',
    15: 'IFOF_L', 16: 'IFOF_R', 17: 'ILF_L', 18: 'ILF_R', 19: 'MCP',
    20: 'MdLF_L', 21: 'MdLF_R', 22: 'OR_ML_L', 23: 'OR_ML_R',
    24: 'POPT_L', 25: 'POPT_R', 26: 'PYT_L', 27: 'PYT_R',
    28: 'SLF_L', 29: 'SLF_R', 30: 'UF_L', 31: 'UF_R'
}

# -------------------- SELECT SUBJECTS --------------------
train_subjects = get_subjects(trainpath)
n_train = int(len(train_subjects) * 0.2)
train_subjects = train_subjects[:n_train]  # Use 20% of subjects for processing

test_subjects = get_subjects(testpath)
n_test = int(len(test_subjects) * 0.2)
test_subjects = test_subjects[:n_test]

# -------------------- PROCESS TRAIN AND TEST --------------------
train_dataset, train_labels = undersample_batches_subjects(bundles_names, trainpath, train_subjects, th, p, batch_size=2)
train_dataset_def, train_labels_def = preprocessing(train_dataset, train_labels)

test_dataset, test_labels = undersample_batches_subjects(bundles_names, testpath, test_subjects, th, p, batch_size=2)
test_dataset_def, test_labels_def = preprocessing(test_dataset, test_labels)

# -------------------- SAVE DATA --------------------
np.save('dataset/base_train_database.npy', train_dataset_def)
np.save('dataset/base_train_labels.npy', train_labels_def)
np.save('dataset/test_database.npy', test_dataset_def)
np.save('dataset/test_labels.npy', test_labels_def)
