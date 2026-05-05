# Credit Card Fraud Detection System

## Overview
This project detects fraudulent credit card transactions using Machine Learning.

## Problem
Fraudulent transactions cause huge financial losses and need real-time detection.

## Solution
We use Random Forest with SMOTE to handle imbalanced data.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Seaborn

## Results
- High Recall for fraud detection
- Balanced classification

## Screenshots
<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5e266a0e-a7c1-4d1b-9027-8d194764e8fa" />
              precision    recall  f1-score   support

           0       1.00      0.99      0.99      4001
           1       0.69      0.92      0.79        98

    accuracy                           0.99      4099
   macro avg       0.84      0.95      0.89      4099
weighted avg       0.99      0.99      0.99      4099

## Dataset

Due to GitHub size limits, the dataset is not included.

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file inside:
data/creditcard.csv

## How to Run
python main.py

## Github repo link:
https://github.com/adityatamhanedev/Credit-Card-Fraud-Detection-System
