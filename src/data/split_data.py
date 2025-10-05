import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    input_path = "data/raw_data/raw.csv"
    output_dir = "data/processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(input_path)

    # Features and target
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save the splits
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("✅ Data split and saved to data/processed_data/")

if __name__ == "__main__":
    main()
