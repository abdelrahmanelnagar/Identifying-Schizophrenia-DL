# Identifying-Schizophrenia-DL
This project aims to identify schizophrenia using a deep learning model trained on resting-state fMRI data from the OpenNeuro dataset.

# Paper Details
Paper Title: Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm

Paper Link: Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm

# Dataset
The dataset is available on OpenNeuro and contains resting-state fMRI data.

The dataset includes structural MRI scans divided into the following groups:

sub-1: Nonsmoking schizophrenia patients.

sub-2: Smoking schizophrenia patients.

sub-3: Nonsmoking healthy controls.

sub-4: Smoking healthy controls.

For this project, we used only the data from sub-1 (nonsmoking schizophrenia patients). The data is organized in NIfTI format and includes metadata for each participant.


Steps to Download the Dataset

1- Visit the dataset link: OpenNeuro Dataset.

2- Click on the "Download Dataset" button.

3- Choose "All Files" or the specific folders you need.

# Repository Structure
notebooks: Contains the main Jupyter notebook for training and evaluation (Identifying-Schizophrenia-Using-DL.ipynb).

models: Saved model architecture and weights.

data: Placeholder for the dataset (not included in the repository; follow the download steps above).

results: Contains visualizations, metrics, and logs from the training process.

# Model Details

Model Architecture:

Input Layer: 3D fMRI data.

Convolutional Layers: 3 layers with ReLU activation.

Pooling Layers: Max pooling for feature reduction.

Dense Layers: Fully connected layers with dropout for regularization.

Output Layer: Softmax for classification (schizophrenia vs. control).

Model Input: Preprocessed 3D fMRI data tensors.

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
