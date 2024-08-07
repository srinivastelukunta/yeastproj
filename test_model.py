import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


@pytest.fixture(scope='module')
def load_model_and_data():
    # Load the saved model and scaler
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load the dataset for testing
    data = pd.read_csv('yeast.csv')
    label_column = 'name'
    label_encoder = LabelEncoder()
    data[label_column] = label_encoder.fit_transform(data[label_column])
    features = data.drop(label_column, axis=1)
    target = data[label_column]

    return model, scaler, features, target


def test_model_shape(load_model_and_data):
    model, scaler, features, target = load_model_and_data
    # Test if the model has the correct shape
    assert features.shape[1] == model.coef_.shape[1], \
        "Feature shape mismatch"


def test_scaler(load_model_and_data):
    model, scaler, features, target = load_model_and_data
    # Test if the scaler transforms the data correctly
    scaled_features = scaler.transform(features)
    assert scaled_features.shape == features.shape, \
        "Scaler output shape mismatch"


def test_model_accuracy(load_model_and_data):
    model, scaler, features, target = load_model_and_data
    # Test if the model accuracy is reasonable
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5, "Model accuracy is too low"

    # Calculate the mean of the predictions without using numpy
    prediction_mean = sum(y_pred) / len(y_pred)
    assert prediction_mean >= 0, "Mean of predictions is negative"
