"""
Reusable prediction function for the student exam-score model.

Example:
    from predict import predict_exam_score

    student = {
        "age": 21,
        "gender": "Female",
        "course": "Mathematics",
        "study_hours": 4.5,
        "class_attendance": 85,
        "internet_access": "Yes",
        "sleep_hours": 7,
        "sleep_quality": "Good",
        "study_method": "Self Study",
        "facility_rating": "Good",
        "exam_difficulty": "Moderate"
    }

    print(predict_exam_score(student))
"""

import pickle
from pathlib import Path

import pandas as pd


MODEL_PATH = Path("models/student_score_model.pkl")


def load_model(model_path=MODEL_PATH):
    """Load the saved model and its fitted DictVectorizer."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Run train.py first."
        )

    with path.open("rb") as file:
        bundle = pickle.load(file)

    return bundle


def predict_exam_score(student_data, model_path=MODEL_PATH):
    """
    Predict the exam score for one student.

    student_data must be a dictionary containing the model input features.
    student_id is not used by the model.
    """
    if not isinstance(student_data, dict):
        raise TypeError("student_data must be a Python dictionary.")

    bundle = load_model(model_path)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    feature_columns = bundle["feature_columns"]

    # Require every feature used during training. This avoids silently
    # treating an omitted numeric value as zero during vectorization.
    missing_fields = [
        column for column in feature_columns
        if column not in student_data or student_data[column] is None
    ]
    if missing_fields:
        raise ValueError(
            "Missing required feature(s): " + ", ".join(missing_fields)
        )

    cleaned_student = {
        column: student_data[column]
        for column in feature_columns
    }
    row = pd.DataFrame([cleaned_student])

    # Apply the same simple missing-value rules used during training.
    for column in row.select_dtypes(include=["number"]).columns:
        # The model expects numeric inputs; reject missing numeric values
        # rather than guessing medians not stored in this simple bundle.
        if row[column].isna().any():
            raise ValueError(f"Missing numeric value for '{column}'.")

    for column in row.select_dtypes(exclude=["number"]).columns:
        row[column] = row[column].fillna("Unknown").astype(str)

    vectorized = vectorizer.transform(row.to_dict(orient="records"))
    prediction = float(model.predict(vectorized)[0])
    return round(prediction, 2)
