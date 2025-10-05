# src/models/search_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def main():
    # Load the scaled training features from a CSV file
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")

    # Load the target variable (labels) and flatten it to a 1D array using .values.ravel()
    # This is needed for scikit-learn models that expect 1D arrays for regression targets
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    # Define a grid of hyperparameters to search over
    # n_estimators: number of trees in the forest
    # max_depth: maximum depth of each tree (None means nodes are expanded until all leaves are pure)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20]
    }

    # Set up GridSearchCV:
    # - RandomForestRegressor with a fixed random_state for reproducibility
    # - param_grid: the dictionary of parameters to test
    # - cv=3: 3-fold cross-validation
    # - scoring="r2": use R² (coefficient of determination) as the evaluation metric
    # - verbose=1: print progress messages to the console
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="r2",
        verbose=1
    )

    # Train the model using grid search to find the best parameters
    grid.fit(X_train, y_train)

    # Save the best model (with best parameters) to a file using joblib
    # The model will be saved as "models/best_model.pkl"
    joblib.dump(grid.best_estimator_, "models/best_model.pkl")

# This ensures the script only runs if executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
