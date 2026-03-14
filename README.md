# Autism Spectrum Disorder (ASD) Detection System using Machine Learning and Deep Learning

## Overview

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that affects communication, behavior, and social interaction. Early identification and analysis of patterns associated with ASD can support better clinical understanding and intervention strategies.

This project explores the application of **Machine Learning and Deep Learning techniques** to analyze neuroimaging quality assessment data and identify patterns that may correlate with ASD-related characteristics.

The system integrates multiple datasets derived from neuroimaging quality metrics and applies a complete machine learning pipeline including **data preprocessing, feature engineering, exploratory data analysis, deep learning model development, and performance evaluation**.

The primary objective of this project is to experiment with different deep learning architectures and analyze their ability to model complex relationships within biomedical datasets.

---

## Problem Statement

Neuroimaging datasets often contain high-dimensional data with complex relationships between features. Traditional statistical approaches may struggle to capture these nonlinear relationships.

This project investigates whether **deep learning models** can effectively learn patterns from combined neuroimaging quality assessment datasets and provide predictive insights.

---

## Dataset Description

The project utilizes three different neuroimaging quality assessment datasets:

| Dataset | Description |
|------|------|
| **anat_qap.csv** | Contains anatomical MRI quality assessment features |
| **dti_qap.csv** | Contains Diffusion Tensor Imaging (DTI) quality metrics |
| **functional_qap.csv** | Contains functional MRI quality assessment features |

Each dataset includes metadata fields such as:

- Site_ID
- Sub_ID
- Session

These attributes are used to construct a **unique merge key**, enabling integration of the three datasets into a unified dataset for analysis.

```
merge_key = Site_ID + "_" + Sub_ID + "_" + Session
```

After merging, the resulting dataset contains a wide range of quantitative neuroimaging quality metrics.

---

## Machine Learning Pipeline

The project implements a structured machine learning workflow consisting of the following stages:

### 1. Data Integration

Multiple datasets are merged into a unified dataset using a custom-generated merge key. This step ensures that data corresponding to the same subject and session is correctly aligned across all imaging modalities.

### 2. Data Preprocessing

Data preprocessing is performed to prepare the dataset for machine learning models. This includes:

- Selecting numerical features
- Handling missing values
- Removing invalid observations
- Standardizing features using **StandardScaler**

Feature normalization is essential to ensure consistent scale across all variables and improve model convergence during training.

---

### 3. Exploratory Data Analysis (EDA)

To understand relationships between variables, correlation analysis is performed across all features.

A **correlation heatmap** is generated using **Seaborn** to visualize relationships among features and identify strongly correlated variables.

This analysis provides insights into potential feature dependencies and helps guide model experimentation.

---

### 4. Feature Transformation

The dataset contains tabular features. To experiment with convolutional neural networks and pretrained architectures, the feature vectors are reshaped into **image-like matrices**.

Steps include:

- Determining an appropriate matrix dimension
- Padding features where necessary
- Reshaping the dataset into a format compatible with CNN-based architectures

This transformation enables the use of deep learning architectures that are typically designed for image data.

---

### 5. Deep Learning Models Implemented

Several neural network architectures are implemented and compared in this project.

#### 1. Dense Neural Network (DNN)

A fully connected feedforward neural network that learns nonlinear relationships between features.

Architecture:
- Flatten Layer
- Dense Layer (ReLU)
- Dense Layer (ReLU)
- Output Layer

---

#### 2. Convolutional Neural Network (CNN)

CNNs are designed to detect spatial patterns and local feature relationships.

Architecture includes:

- Convolution layers
- Max pooling
- Global average pooling
- Dense output layer

---

#### 3. VGG16-based Architecture

A modified implementation of the **VGG16 architecture** is used to experiment with deeper convolutional representations.

The model includes:

- Input reshaping
- Channel expansion
- Resizing to standard CNN input format
- Global pooling layer
- Regression output layer

---

#### 4. ResNet50-based Architecture

ResNet introduces **residual connections**, allowing deeper neural networks to train more effectively.

This project adapts ResNet50 for feature learning from transformed tabular data.

---

#### 5. Transformer-based Model (Experimental)

A simplified transformer architecture using **Multi-Head Attention** is implemented to explore the potential of attention mechanisms in learning feature relationships.

Transformer components used:

- Multi-head attention
- Layer normalization
- Dense feedforward layers

---

## Model Evaluation Metrics

To evaluate model performance, the following regression metrics are used:

| Metric | Description |
|------|------|
| **R² Score** | Measures the proportion of variance explained by the model |
| **Mean Absolute Error (MAE)** | Measures average prediction error |
| **Root Mean Squared Error (RMSE)** | Measures prediction error magnitude |

Additionally, **prediction scatter plots** are generated to compare predicted values against actual values.

---

## Technologies Used

### Programming Language
- Python

### Machine Learning Libraries
- Scikit-learn
- TensorFlow
- Keras

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

---

## Project Structure

```
ASD-Detection-System
│
├── data
│   ├── anat_qap.csv
│   ├── dti_qap.csv
│   └── functional_qap.csv
│
├── ASD_Detection_Model.ipynb
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/shashwatkul/ASD-Detection-System.git
```

Navigate to the project directory:

```
cd ASD-Detection-System
```

Install required dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

You can run the project in two ways:

### Option 1: Python Script

```
python train_model.py
```

### Option 2: Jupyter Notebook

Open and run:

```
ASD_Detection_Model.ipynb
```

---

## Future Improvements

Several improvements can further enhance the system:

- Hyperparameter tuning using GridSearch or Bayesian optimization
- Feature importance analysis using **SHAP**
- Model interpretability techniques
- Cross-validation experiments
- Deployment using Flask or FastAPI
- Integration with an interactive dashboard

---

## Author

**Shashwat Kulshrestha**

B.Tech – Information Technology  
Aspiring Artificial Intelligence / Machine Learning Engineer

---

## License

This project is intended for research and educational purposes.
