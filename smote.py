import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer

# -------------------- LOAD DATA --------------------
X_data = np.load('dataset/base_train_database.npy')  # Training dataset (N, 15, 3)
y_data = np.load('dataset/base_train_labels.npy')    # One-hot labels (N, 32)

# -------------------- ONE-HOT TO CLASS INDEX --------------------
def one_hot_to_label(y_onehot):
    return np.argmax(y_onehot, axis=1)

y_data_num = one_hot_to_label(y_data)

# -------------------- FLATTEN FOR SMOTE --------------------
N, n_points, coords = X_data.shape   # (N, 15, 3)
X_data_flat = X_data.reshape((N, -1))  # (N, 15*3 = 45)

# -------------------- APPLY SMOTE --------------------
# Oversample only classes 13 and 14 until 20,000 samples each
target_counts = {13: 20000, 14: 20000}
smote = SMOTE(sampling_strategy=target_counts, random_state=42)
X_resampled_flat, y_resampled = smote.fit_resample(X_data_flat, y_data_num)

# -------------------- RESHAPE BACK --------------------
X_resampled = X_resampled_flat.reshape((-1, n_points, coords))  # (N_new, 15, 3)

# -------------------- CONVERT BACK TO ONE-HOT --------------------
lb = LabelBinarizer()
lb.fit(np.arange(32))  # Ensure we always have 32 classes
y_resampled_onehot = lb.transform(y_resampled)  # (N_new, 32)

# -------------------- SAVE OUTPUT --------------------
np.save("dataset/smote_train_database.npy", X_resampled)
np.save("dataset/smote_train_labels.npy", y_resampled_onehot)

print("Final dataset:", X_resampled.shape, y_resampled_onehot.shape)
