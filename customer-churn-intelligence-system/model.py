import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

DATA_PATH = "data/customer_data.csv"
MODEL_PATH = "models/churn_model.pkl"


def train_model():
    data = pd.read_csv(DATA_PATH)

    X = data[["usage_hours", "complaints"]]
    y = data["churn"]

    model = LogisticRegression()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ Customer Churn model trained and saved successfully")


def load_model():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_model()
