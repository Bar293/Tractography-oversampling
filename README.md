# Synthetic Data Generation for Brain Tractography Segmentation

This repository contains the code for my Master's Final Project (TFM). The project investigates the impact of different oversampling techniques on the performance of a convolutional neural network (CNN) model for tractography bundle classification.

## Project Files

The main files included in this repository are:

- **requirements.txt** – List of Python libraries required to run the project.
- **make_dataset.py** – Extracts the streamlines of different bundles from the subjects of the [Tractoinferno dataset](https://openneuro.org/datasets/ds003900/versions/1.1.1).
- **mni.py** – Registers the tractography data to the MNI space with a common voxel size of 1 mm.
- **fascicle_data.py** – Collects all streamlines from the original Tractoinferno dataset. Used to obtain all samples from the underrepresented bundles (FX_L and FX_R).
- **fiber_count.py** – Plots the number of streamlines per bundle in a given dataset split.
- **basic.py** – Performs oversampling of underrepresented bundles by randomly duplicating existing samples.
- **smote.py** – Balances the dataset using the SMOTE technique.
- **vae.py** – Trains a Variational Autoencoder (VAE) for each underrepresented bundle and generates synthetic samples to balance the dataset.
- **model.py** – Trains the CNN model with different datasets and computes metrics including accuracy, precision, recall, F1-score, and a normalized confusion matrix.
- **map.py** – Visualizes augmented samples in a 2D space.

## Results

The table below summarizes the performance comparison between different oversampling techniques:

| Method      | FX_L Precision | FX_R Precision | FX_L Recall | FX_R Recall | FX_L F1  | FX_R F1  | Observations                                                                                                                     |
| ----------- | -------------- | -------------- | ----------- | ----------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Baseline    | 65.8           | **100**        | 93.5        | 52.3        | 77.2     | 68.7     | High precision but poor recall for FX_R; clear under-representation effects visible in both fornix bundles.                      |
| Duplication | 49.3           | 90.7           | 97.2        | 82.2        | 65.4     | 86.3     | Strong recall boost but precision drops, indicating overfitting and minority-label overprediction.                               |
| VAE         | **88.4**       | **100**        | 92.5        | 86.0        | **90.4** | **92.5** | Best precision–recall balance; synthetic samples add meaningful variability but may shift the feature space for complex bundles. |
| SMOTE       | 78.4           | 86.2           | **98.1**    | **93.5**    | 87.1     | 89.7     | Highest recall; expands the decision boundary effectively but increases false positives relative to VAE.                         |


## Full Work

The complete Master's thesis can be accessed at the following link:  
[http://hdl.handle.net/10045/159604](http://hdl.handle.net/10045/159604)
