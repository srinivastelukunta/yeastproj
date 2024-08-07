import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from pickle import dump

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('yeast.csv')

    # The label column name is 'name'
    label_column = 'name'

    # Data preprocessing
    label_encoder = LabelEncoder()
    data[label_column] = label_encoder.fit_transform(data[label_column])
    features = data.drop(label_column, axis=1)
    target = data[label_column]

    return features, target

# Train and evaluate the model
def train_and_evaluate_model(features, target):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return model, scaler

# Save model and Scaler
features, target = load_and_preprocess_data()
model, scaler = train_and_evaluate_model(features, target)
pickle.dump(model, open('model.pkl', 'wb'))
dump(scaler, open('scaler.pkl', 'wb'))
