# Identifying Schizophrenia Using Deep Learning and fMRI Data

This repository contains code for identifying schizophrenia using a deep learning model trained on resting-state fMRI data from the OpenNeuro dataset.

## Paper Details
- **Paper Title**: Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm
- **Paper Link**: [Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm](https://example.com) *(replace with actual link)*

---

## Model Details

### Architecture
The model is a 3D Convolutional Neural Network (CNN) with the following architecture:
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

---

## Dataset
The dataset is available on [OpenNeuro](https://openneuro.org) and contains resting-state fMRI data. It includes structural MRI scans divided into the following groups:
- **sub-1**: Nonsmoking schizophrenia patients.
- **sub-2**: Smoking schizophrenia patients.
- **sub-3**: Nonsmoking healthy controls.
- **sub-4**: Smoking healthy controls.

For this project, we used only the data from **sub-1** (nonsmoking schizophrenia patients) and **sub-3** (nonsmoking healthy controls). The data is organized in NIfTI format and includes metadata for each participant.

### Steps to Download the Dataset
1. Visit the dataset link: [OpenNeuro Dataset](https://openneuro.org).
2. Click on the "Download Dataset" button.
3. Choose "All Files" or the specific folders you need.

## Repository Structure
Identifying-Schizophrenia-DL/
│
├── notebooks/
│ └── Identifying-Schizophrenia-Using-DL.ipynb # Main Jupyter notebook for training and evaluation
│
├── models/
│ └── (Saved model architecture and weights)
│
├── data/
│ └── (Placeholder for the dataset; follow the download steps above)
│
├── results/
│ └── (Visualizations, metrics, and logs from the training process)
│
├── README.md # This file
├── requirements.txt # List of dependencies
└── LICENSE # License file

# Dependencies

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
