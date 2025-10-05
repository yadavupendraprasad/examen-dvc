import pandas as pd
import joblib

def main():
    model = joblib.load("models/best_model.pkl")
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    model.fit(X_train, y_train)
    joblib.dump(model, "models/final_model.pkl")

if __name__ == "__main__":
    main()
