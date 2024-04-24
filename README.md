# Network Anomaly Detection using Machine Learning

This repository contains a Python script for network anomaly detection using various machine learning algorithms. The script preprocesses the data, builds different models, and evaluates their performance.
The highest Accuracy reached is 89% for test data prediction.

## Dataset
The dataset used in this project contains network traffic data with features such as duration, protocol type, service, flag, etc. The dataset is divided into training and testing sets.

- Training Dataset: `Train.txt`
- Testing Dataset: `Test.txt`

## Dependencies
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - tensorflow

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/network-anomaly-detection.git
2. Navigate to the project directory:
   ```bash
   cd network-anomaly-detection

## Models Used
The script implements the following machine learning models:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Neural Network

## Results
The script evaluates each model's performance on both the training and testing datasets, reporting metrics such as accuracy, precision, recall, F1-score, and ROC curve

## Folder structure :
    ```
    network-anomaly-detection/
    │
    ├── data/
    │   ├── Train.txt
    │   └── Test.txt
    │
    ├── network_anomaly_detection.py
    └── README.md

## Licence :
This project is licensed under the MIT License.
