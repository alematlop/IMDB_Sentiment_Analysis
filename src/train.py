import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from preprocess import clean_text
import pandas as pd
import numpy as np


# Load & preprocess data

df_train = pd.read_csv('../data/processed/train.csv')

df_train['text'] = df_train['text'].apply(clean_text)


# Convert text data to numerical features

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(df_train['text'])
Y_train = df_train['label']

# Logistic Regression model training

model = LogisticRegression()
model.fit(X_train, Y_train)

joblib.dump(model, "../models/logistic_regression.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
print("Model training complete and saved!")