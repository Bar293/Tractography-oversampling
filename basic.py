import numpy as np

def oversample_bundle(dataset_path, labels_path, class_index, target, 
                      output_data_path, output_labels_path):
    """
    Oversample a specific bundle by repeating its samples until the target number is reached,
    then concatenate with the base dataset.
    
    Args:
        dataset_path (str): path to the base dataset (.npy) with shape (N, 15, 3)
        labels_path (str): path to the base labels (.npy) with one-hot encoding
        class_index (int): index of the class to oversample
        target (int): total desired number of samples for that class
        output_data_path (str): path to save the augmented dataset
        output_labels_path (str): path to save the augmented labels
    """

    # -------------------- LOAD DATA --------------------
    X = np.load(dataset_path)  # (N, 15, 3)
    y = np.load(labels_path)   # (N, num_classes)
    num_classes = y.shape[1]

    # -------------------- FILTER TARGET CLASS --------------------
    y_classes = np.argmax(y, axis=1)
    bundle_samples = X[y_classes == class_index]
    bundle_count = bundle_samples.shape[0]
    print(f"Class {class_index}: {bundle_count} current samples.")

    if bundle_count == 0:
        raise ValueError(f"No samples found for class {class_index} in dataset.")

    # -------------------- CHECK IF OVERSAMPLING IS NEEDED --------------------
    to_generate = target - bundle_count
    if to_generate <= 0:
        print(f"Already {bundle_count} samples >= target {target}. Nothing to do.")
        np.save(output_data_path, X)
        np.save(output_labels_path, y)
        return

    # -------------------- CREATE SYNTHETIC SAMPLES --------------------
    indices = np.random.choice(bundle_count, size=to_generate, replace=True)
    synthetic_samples = bundle_samples[indices]

    # Create corresponding one-hot labels
    synthetic_labels = np.zeros((to_generate, num_classes))
    synthetic_labels[:, class_index] = 1

    # -------------------- CONCATENATE DATASETS --------------------
    X_new = np.concatenate([X, synthetic_samples], axis=0)
    y_new = np.concatenate([y, synthetic_labels], axis=0)

    # -------------------- SAVE OUTPUT --------------------
    np.save(output_data_path, X_new)
    np.save(output_labels_path, y_new)

    print(f"Augmented dataset saved at {output_data_path}, {output_labels_path}")
    print(f"New dataset: {X_new.shape}, labels: {y_new.shape}")


# -------------------- EXECUTION --------------------
oversample_bundle(
    dataset_path="dataset/base_train_database.npy",
    labels_path="dataset/base_train_labels.npy",
    class_index=13,  # FX_L
    target=20000,
    output_data_path="dataset/copy_train_database.npy",
    output_labels_path="dataset/copy_train_labels.npy"
)

oversample_bundle(
    dataset_path="dataset/copy_train_database.npy",
    labels_path="dataset/copy_train_labels.npy",
    class_index=14,  # FX_R
    target=20000,
    output_data_path="dataset/copy_train_database.npy",
    output_labels_path="dataset/copy_train_labels.npy"
)
