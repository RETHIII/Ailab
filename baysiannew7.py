import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

# Load dataset
heartDisease = pd.read_csv('7.csv')
heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dropna(inplace=True)  # Drop rows with missing values

# Display sample instances from the dataset
print('Sample instances from the dataset are given below')
print(heartDisease.head())

# Display attributes and their data types
print('\nAttributes and datatypes')
print(heartDisease.dtypes)

# Define the features and target variable
features = ['age', 'gender', 'exang', 'cp', 'restecg', 'chol']
X = heartDisease[features]
y = heartDisease['heartdisease']

# Split the data into training and testing sets (if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes model
model = CategoricalNB()

# Train the model
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(X_train, y_train)

# Perform inference with the trained model
print('\nInferencing with Bayesian Network:')

# Query 1: Probability of HeartDisease given evidence= restecg
print('\n1. Probability of HeartDisease given evidence= restecg')
restecg_evidence = np.array([[63, 1, 0, 3, 1, 233]])  # Example evidence with restecg=1
proba_restecg = model.predict_proba(restecg_evidence)
print(f"+-----------------+---------------------+")
print(f"| heartdisease    |   phi(heartdisease) |")
print(f"+=================+=====================+")
print(f"| heartdisease(0) |              {proba_restecg[0][0]:.4f} |")
print(f"| heartdisease(1) |              {proba_restecg[0][1]:.4f} |")
print(f"+-----------------+---------------------+")

# Query 2: Probability of HeartDisease given evidence= cp
print('\n2. Probability of HeartDisease given evidence= cp')
cp_evidence = np.array([[63, 1, 0, 2, 0, 233]])  # Example evidence with cp=2
proba_cp = model.predict_proba(cp_evidence)
print(f"+-----------------+---------------------+")
print(f"| heartdisease    |   phi(heartdisease) |")
print(f"+=================+=====================+")
print(f"| heartdisease(0) |              {proba_cp[0][0]:.4f} |")
print(f"| heartdisease(1) |              {proba_cp[0][1]:.4f} |")
print(f"+-----------------+---------------------+")
