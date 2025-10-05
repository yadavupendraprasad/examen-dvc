import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")
    
    # Drop datetime or any non-numeric columns before scaling
    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test.select_dtypes(include=["number"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed_data/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/processed_data/X_test_scaled.csv", index=False)

if __name__ == "__main__":
    main()
