"""
Train a regression model to predict student exam scores.

Expected CSV columns:
student_id, age, gender, course, study_hours, class_attendance,
internet_access, sleep_hours, sleep_quality, study_method,
facility_rating, exam_difficulty, exam_score

Run:
    python train.py --data data/student_performance.csv

The script saves a bundle containing both the DictVectorizer and model
to models/student_score_model.pkl.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TARGET = "exam_score"
ID_COLUMNS = ["student_id"]
MODEL_PATH = Path("models/student_score_model.pkl")


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Remove target/identifier columns and handle missing feature values."""
    features = data.drop(columns=[TARGET] + ID_COLUMNS, errors="ignore").copy()

    # Fill missing numeric values with the column median.
    numeric_columns = features.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        median = features[column].median()
        features[column] = features[column].fillna(median)

    # Fill missing categorical values with the most frequent value,
    # or 'Unknown' if a column contains no non-missing values.
    categorical_columns = features.select_dtypes(exclude=["number"]).columns
    for column in categorical_columns:
        mode = features[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        features[column] = features[column].fillna(fill_value).astype(str)

    return features


def evaluate(model, X, y, label):
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print(f"\n{label} results")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/student_performance.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--model-output",
        default=str(MODEL_PATH),
        help="Where to save the trained model bundle",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data)

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.dropna(subset=[TARGET]).copy()
    y = data[TARGET].astype(float)
    X = prepare_features(data)

    # 60% training, 20% validation, 20% test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )

    # Fit the vectorizer on training data only.
    vectorizer = DictVectorizer(sparse=False)
    X_train_dict = X_train.to_dict(orient="records")
    X_val_dict = X_val.to_dict(orient="records")
    X_test_dict = X_test.to_dict(orient="records")

    X_train_vectorized = vectorizer.fit_transform(X_train_dict)
    X_val_vectorized = vectorizer.transform(X_val_dict)
    X_test_vectorized = vectorizer.transform(X_test_dict)

    # Linear Regression was the best baseline model in the validation
    # comparison previously shared for this project.
    model = LinearRegression()
    model.fit(X_train_vectorized, y_train)

    print("Validation performance:")
    evaluate(model, X_val_vectorized, y_val, "Validation")

    # After model selection, refit on train + validation.
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_vectorizer = DictVectorizer(sparse=False)
    X_train_full_vectorized = final_vectorizer.fit_transform(
        X_train_full.to_dict(orient="records")
    )

    final_model = LinearRegression()
    final_model.fit(X_train_full_vectorized, y_train_full)

    # Evaluate the final model once on the held-out test set.
    X_test_final = final_vectorizer.transform(X_test_dict)
    test_metrics = evaluate(final_model, X_test_final, y_test, "Test")

    model_bundle = {
        "model": final_model,
        "vectorizer": final_vectorizer,
        "feature_columns": list(X_train_full.columns),
        "target": TARGET,
        "dropped_columns": ID_COLUMNS,
    }

    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(model_bundle, file)

    print(f"\nSaved model bundle to: {output_path}")
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
