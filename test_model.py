import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@pytest.fixture(scope='module')
def load_data():
    # Load the dataset for testing
    data = pd.read_csv('yeast.csv')
    label_column = 'name'
    label_encoder = LabelEncoder()
    data[label_column] = label_encoder.fit_transform(data[label_column])
    features = data.drop(label_column, axis=1)
    target = data[label_column]

    return features, target


def test_model_accuracy(load_data):
    features, target = load_data
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    # Standardizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5, "Model accuracy is too low"
    # Ensure predictions are non-negative
    assert all(pred >= 0 for pred in y_pred), "Some predictions are negative"
