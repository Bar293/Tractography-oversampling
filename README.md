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

| Method                     | Train Acc. | Test Acc. | FX_L Acc. | FX_L F1  | FX_R Acc. | FX_R F1  | Observations                                                  |
| -------------------------- | ---------- | --------- | --------- | -------- | --------- | -------- | ------------------------------------------------------------- |
| Baseline (no oversampling) | 0.91       | 0.83      | 0.42      | 0.38     | 0.45      | 0.40     | Strong bias toward majority bundles; fornix poorly classified |
| Duplication                | 0.96       | 0.84      | 0.55      | 0.50     | 0.53      | 0.48     | Improves minority classes but risk of overfitting             |
| VAE                        | 0.93       | 0.85      | 0.62      | 0.58     | 0.60      | 0.55     | Synthetic samples increase diversity; moderate gains          |
| SMOTE                      | 0.94       | **0.88**  | **0.71**  | **0.68** | **0.73**  | **0.69** | Best trade-off; consistent improvements                       |

## Full Work

The complete Master's thesis can be accessed at the following link:  
[http://hdl.handle.net/10045/159604](http://hdl.handle.net/10045/159604)
