# Credit Card Fraud Detection

## Overview
This project implements a machine learning solution for detecting fraudulent credit card transactions using Linear Regression. The model analyzes transaction patterns to classify transactions as either fraudulent (1) or legitimate (0).

## Dataset
The dataset used in this project is from Kaggle's Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset features include:

- `Time`: Number of seconds elapsed between this transaction and the first transaction
- `Amount`: Transaction amount
- `V1-V28`: Principal components obtained with PCA transformation (anonymized features)
- `Class`: Target variable (1 for fraud, 0 for legitimate)

## Requirements
```
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
matplotlib==3.4.2
```

## Usage
1. Data Preprocessing:
   ```python
   from src.data_preprocessing import preprocess_data
   
   # Load and preprocess the data
   X_train, X_test, y_train, y_test = preprocess_data('data/creditcard.csv')
   ```

2. Model Training:
   ```python
   from src.model import train_model
   
   # Train the linear regression model
   model = train_model(X_train, y_train)
   ```

3. Making Predictions:
   ```python
   # Predict on new data
   predictions = model.predict(X_test)
   ```

## Model Performance
The Linear Regression model's performance metrics:
- Accuracy on Training data :  0.9453621346886912
- Accuracy score on Test Data : 0.9086294416243654
