 
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator # Now you can import from pgmpy
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Load dataset
heartDisease = pd.read_csv('7.csv')
heartDisease = heartDisease.replace('?', np.nan)

# Display sample instances from the dataset
print('Sample instances from the dataset are given below')
print(heartDisease.head())

# Display attributes and their data types
print('\n Attributes and datatypes')
print(heartDisease.dtypes)

# Define the Bayesian Network structure
model = BayesianModel([
    ('age', 'heartdisease'), 
    ('gender', 'heartdisease'), 
    ('exang', 'heartdisease'), 
    ('cp', 'heartdisease'), 
    ('heartdisease', 'restecg'), 
    ('heartdisease', 'chol')
])

# Learn CPD using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Perform inference with the Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)

# Query 1: Probability of HeartDisease given evidence= restecg
print('\n1. Probability of HeartDisease given evidence= restecg')
q1 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

# Query 2: Probability of HeartDisease given evidence= cp
print('\n2. Probability of HeartDisease given evidence= cp')
q2 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)