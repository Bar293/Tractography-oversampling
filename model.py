import psutil
import humanize
import os
import GPUtil as GPU
import numpy as np
import time
import math 
import random
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

print("TensorFlow version: " + tensorflow.__version__)


# -------------------- GPU MONITORING --------------------
tensorflow.test.gpu_device_name()
GPUs = GPU.getGPUs()
gpu = GPUs[0]

def printm():
    """Prints memory usage for CPU and GPU."""
    process = psutil.Process(os.getpid())
    print("General RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
          " | Process size: " + humanize.naturalsize(process.memory_info().rss))
    c = "GPU RAM Free: {0:.0f} MB | Used: {1:.0f} MB | Util: {2:3.0f} % | Total: {3:.0f} MB"
    print(c.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal))


# -------------------- LOAD DATA --------------------
train_attributes = np.load('dataset/base_train_database.npy')
train_labels = np.load('dataset/base_train_labels.npy')
test_attributes = np.load('dataset/test_database.npy')
test_labels = np.load('dataset/test_labels.npy')
# train_attributes = np.load('dataset/vae_train_database.npy')
# train_labels = np.load('dataset/vae_train_labels.npy')
# train_attributes = np.load('dataset/smote_train_database.npy')
# train_labels = np.load('dataset/smote_train_labels.npy')
# train_attributes = np.load('dataset/copy_train_database.npy')
# train_labels = np.load('dataset/copy_train_labels.npy')

print(train_attributes.shape, type(train_attributes))
print(train_labels.shape, type(train_labels))
print(test_attributes.shape, type(test_attributes))
print(test_labels.shape, type(test_labels))


# -------------------- DATA GENERATOR --------------------
class DataGenerator(tensorflow.keras.utils.Sequence):
    """Generates batches of FiberMap images for training."""

    def __init__(self, dataset, labels, batch_siz=256, p=15, shuffle=True):
        self.dataset = dataset
        self.labels = labels
        self.batch_siz = batch_siz
        self.p = p
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(math.ceil(len(self.dataset) / self.batch_siz))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_siz: (index + 1) * self.batch_siz]
        x, y = [], []
        for i in indexes:
            x.append(self.FiberMap(self.dataset[i]))
            y.append(self.labels[i])
        x, y = np.array(x), np.array(y)
        x = x.transpose(0, 2, 3, 1).astype('float32')  # shape (N, H, W, C)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def FiberMap(self, ST):
        ST = ST.T
        R = self.sub_FiberMap(ST[0])
        G = self.sub_FiberMap(ST[1])
        B = self.sub_FiberMap(ST[2])
        return np.array([R, G, B])
    
    def sub_FiberMap(self, line):
        line = line.tolist()
        line_flipped = line[::-1]
        block = [line + line_flipped, line_flipped + line]
        return np.array(block * self.p)


# -------------------- DATA SPLIT --------------------
def make_table(y, fascicles):
    """Prints a table with the number of examples per class."""
    y_to_int = np.argmax(y, axis=-1).tolist()
    labels_occs = Counter(y_to_int)
    data = [labels_occs.get(i, 0) for i in range(len(fascicles))]

    y_table = pd.DataFrame(
        [fascicles[:16], data[:16], fascicles[16:], data[16:]],
        columns=['-'] * 16
    )
    print(y_table)


fascicles = ['AF_L', 'AF_R', 'CC_Fr_1', 'CC_Fr_2', 'CC_Oc', 'CC_Pa', 'CC_Pr_Po', 'CG_L', 'CG_R',
             'FAT_L', 'FAT_R', 'FPT_L', 'FPT_R', 'FX_L', 'FX_R', 'IFOF_L', 'IFOF_R', 'ILF_L', 'ILF_R',
             'MCP', 'MdLF_L', 'MdLF_R', 'OR_ML_L', 'OR_ML_R', 'POPT_L', 'POPT_R', 'PYT_L', 'PYT_R',
             'SLF_L', 'SLF_R', 'UF_L', 'UF_R']

train_ratio = 0.8
n_instances = train_attributes.shape[0]
n_train = int(n_instances * train_ratio)

# Shuffle training and testing sets
train_perm = np.random.permutation(n_instances)
test_perm = np.random.permutation(test_attributes.shape[0])

train_attributes = train_attributes[train_perm]
train_labels = train_labels[train_perm]
test_attributes = test_attributes[test_perm]
test_labels = test_labels[test_perm]

x_train = train_attributes[:n_train]
y_train = train_labels[:n_train]
x_dev = train_attributes[n_train:]
y_dev = train_labels[n_train:]
x_test = test_attributes
y_test = test_labels

num_train_examples = x_train.shape[0]
num_dev_examples = x_dev.shape[0]
num_test_examples = x_test.shape[0]

print("Data dimensions: ", x_train.shape)
print("Training examples: ", num_train_examples)
print("Development examples: ", num_dev_examples)
print("Test examples: ", num_test_examples)
print("Training set distribution:")
make_table(y_train, fascicles)
print("Development set distribution:")
make_table(y_dev, fascicles)
print("Test set distribution:")
make_table(y_test, fascicles)


# -------------------- TRAINING CONFIG --------------------
learning_rate = 0.001
n_epochs = 30
batch_size = 512

training_generator = DataGenerator(x_train, y_train, batch_siz=batch_size)
development_generator = DataGenerator(x_dev, y_dev, batch_siz=batch_size)
validation_generator = DataGenerator(x_test, y_test, batch_siz=batch_size)


# -------------------- MODEL DEFINITION --------------------
pad = 'same'
kernel_init = 'glorot_uniform' 
learning_rate = 0.001

# Model
model = Sequential()

# 1st convolutional layer
model.add(keras.Input(shape=(30, 30, 3)))
model.add(Conv2D(filters= 16, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 2nd convolutional layer
model.add(Conv2D(filters= 16, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 3rd convolutional layer
model.add(Conv2D(filters= 32, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 4th convolutional layer
model.add(Conv2D(filters= 32, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 5th convolutional layer
model.add(Conv2D(filters= 64, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 6th convolutional layer
model.add(Conv2D(filters= 64, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 7th convolutional layer
model.add(Conv2D(filters= 128, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# 8th convolutional layer
model.add(Conv2D(filters= 128, kernel_size= (3, 3), strides= (1, 1), padding= pad, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2), strides= (1, 1), padding= pad))
model.add(Dropout(0.025))

# Passing it to a Fully Connected layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(64, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(Dropout(0.002))

# 2nd Fully Connected Layer
model.add(Dense(128, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(Dropout(0.002))

# 3rd Fully Connected Layer
model.add(Dense(256, kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(Dropout(0.002))

# Output Layer
model.add(Dense(32, kernel_initializer= kernel_init))
model.add(Activation('softmax'))


model.compile(optimizer= Adam(learning_rate= learning_rate), loss= 'categorical_crossentropy', metrics= ['accuracy'])
print("Modelo resumido:")
model.summary()


# -------------------- PLOTTING FUNCTIONS --------------------
def plot_model_history(model_history, filename='model_history.png'):
    """Plots training/validation accuracy and loss over epochs."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Accuracy
    axs[0].plot(model_history.history['accuracy'])
    axs[0].plot(model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='best')
    # Loss
    axs[1].plot(model_history.history['loss'])
    axs[1].plot(model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='best')
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.colormaps.get_cmap("Blues"),
                          filename='results/confusion_matrix.png'):
    """Plots and saves a confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.array([[round(elem, 5) for elem in row] for row in cm])

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f') if normalize else int(cm[i, j]),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()
    return cm


# -------------------- TRAINING --------------------
start = time.time()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

y_to_int = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_to_int), y=y_to_int)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    training_generator,
    validation_data=development_generator,
    epochs=n_epochs,
    class_weight=class_weights,
    verbose=1,
    callbacks=[early_stopping]
)

end = time.time()
test_loss, test_acc = model.evaluate(validation_generator, verbose=1)

err_train = round((1 - history.history['accuracy'][-1]) * 100, 1)
err_dev = round((1 - history.history['val_accuracy'][-1]) * 100, 1)
bias = err_train
variance = round(err_dev - err_train, 2)

print('Train error:', err_train)
print('Development error:', err_dev)
print('Bias:', bias)
print('Variance:', variance)
print('Train accuracy:', 100 - err_train)
print('Development accuracy:', 100 - err_dev)
print('Test accuracy:', test_acc * 100)
print("Training took " + str(end - start) + " seconds")

plot_model_history(history)


# -------------------- CONFUSION MATRIX & METRICS --------------------
validation_generator = DataGenerator(x_test, y_test, batch_siz=batch_size, p=15, shuffle=False)
y_pred = model.predict(validation_generator, verbose=1)

y_test_cm = np.argmax(y_test, axis=-1).tolist()
y_pred_cm = np.argmax(y_pred, axis=-1).tolist()
acc_predict = sum([int(y_test_cm[i] == y_pred_cm[i]) for i in range(len(y_pred))]) * 100 / len(y_pred)

print('Prediction accuracy:', acc_predict)
print(y_test_cm[:10], '.........')
print(y_pred_cm[:10], '.........')
print(y_test.shape, y_pred.shape)

print("\nClassification report (test):")
print(classification_report(y_test_cm, y_pred_cm, target_names=fascicles, digits=3))

precision = precision_score(y_test_cm, y_pred_cm, average='weighted', zero_division=0)
recall = recall_score(y_test_cm, y_pred_cm, average='weighted', zero_division=0)
f1 = f1_score(y_test_cm, y_pred_cm, average='weighted', zero_division=0)
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

y_test_cm_arr = np.array(y_test_cm[:len(y_pred_cm)])
y_pred_cm_arr = np.array(y_pred_cm)
cnf_matrix = confusion_matrix(y_test_cm_arr, y_pred_cm_arr)

plt.figure(figsize=(35, 35))
cm = plot_confusion_matrix(cnf_matrix, classes=fascicles, normalize=True,
                           title='Normalized confusion matrix')

accs, confs = [], []
for i in range(len(cm)):
    acc = cm[i][i]
    accs.append(acc)
    if acc <= 0.95:
        row = cm[i].tolist()
        row[i] = 0
        new_max = max(row)
        confs.append(fascicles[row.index(new_max)] + ': ' + str(new_max))
    else:
        confs.append(None)
