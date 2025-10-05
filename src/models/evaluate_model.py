import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def main():
    model = joblib.load("models/final_model.pkl")
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()
 
    predictions = model.predict(X_test)
    pd.DataFrame({"Predicted": predictions}).to_csv("data/predictions.csv", index=False)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)

if __name__ == "__main__":
    main()


