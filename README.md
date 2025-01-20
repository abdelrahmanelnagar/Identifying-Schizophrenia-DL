# Identifying-Schizophrenia-DL
This project aims to identify schizophrenia using a deep learning model trained on resting-state fMRI data from the OpenNeuro dataset.

# Paper Details
Paper Title: Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm

Paper Link: [Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2020.00016/full)

# Dataset
The dataset is available on OpenNeuro and contains resting-state fMRI data.

The dataset includes structural MRI scans divided into the following groups:

sub-1: Nonsmoking schizophrenia patients.

sub-2: Smoking schizophrenia patients.

sub-3: Nonsmoking healthy controls.

sub-4: Smoking healthy controls.

For this project, we used only the data from sub-1 and sub-3 (nonsmoking schizophrenia patients and nonsmoking healthy control). The data is organized in NIfTI format and includes metadata for each participant.


Steps to Download the Dataset

1- Visit the dataset link: [OpenNeuro Dataset.](https://openneuro.org/datasets/ds001461/versions/1.0.3)

2- Click on the "Download Dataset" button.

3- Choose "All Files" or the specific folders you need.

# Repository Structure
notebooks: Contains the main Jupyter notebook for training and evaluation (Identifying-Schizophrenia-Using-DL.ipynb).

models: Saved model architecture and weights.

data: Placeholder for the dataset (not included in the repository; follow the download steps above).

results: Contains visualizations, metrics, and logs from the training process.

## Model Details

### Architecture
The model is a 3D CNN with the following architecture:
- **Input Shape**: `(32, 32, 16, 120)` (Depth, Height, Width, Time)
- **Layers**:
  1. Conv3D (16 filters, kernel size `(3, 3, 3)`, ReLU activation)
  2. MaxPooling3D (pool size `(2, 2, 2)`)
  3. BatchNormalization
  4. Dropout (0.25)
  5. Conv3D (32 filters, kernel size `(3, 3, 3)`, ReLU activation)
  6. MaxPooling3D (pool size `(2, 2, 2)`)
  7. BatchNormalization
  8. Dropout (0.25)
  9. Flatten
  10. Dense (128 units, ReLU activation, L2 regularization)
  11. Dropout (0.5)
  12. Dense (1 unit, sigmoid activation)

### Training Details
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy
- **Batch Size**: 8
- **Epochs**: 50
- **Callbacks**:
  - ReduceLROnPlateau (factor=0.2, patience=5, min_lr=0.001)
  - EarlyStopping (patience=10, restore_best_weights=True)

### Performance
- **Accuracy**: 75%
- **Precision**:
  - Healthy: 0.67
  - Diseased: 0.80
- **Recall**:
  - Healthy: 0.67
  - Diseased: 0.80
- **F1-Score**:
  - Healthy: 0.67
  - Diseased: 0.80

## Dataset
The dataset contains fMRI data from:
- **Healthy Controls**: Subjects 301 to 312
- **Diseased Subjects**: Subjects 101 to 125

# Dependencies
Python Version

Python 3.8+

Required Libraries

TensorFlow 2.6.0

NumPy 1.21.0

Pandas 1.3.0

Matplotlib 3.4.2

Scikit-learn 0.24.2

Nibabel 3.2.1

Installation Commands

Run the following command to install the required dependencies:

pip install tensorflow==2.6.0 numpy==1.21.0 pandas==1.3.0 matplotlib==3.4.2 scikit-learn==0.24.2 nibabel==3.2.1
