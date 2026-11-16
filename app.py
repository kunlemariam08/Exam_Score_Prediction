"""
Flask API for student exam-score prediction.

Start:
    python app.py

Test:
    POST http://127.0.0.1:5000/predict
    JSON body: see example_payload.json or the README instructions.
"""

from flask import Flask, jsonify, request

from predict import predict_exam_score

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "message": "Student Exam Score Prediction API is running.",
        "endpoint": "POST /predict",
        "input": "Send student feature values as a JSON object."
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    student_data = request.get_json(silent=True)

    if not isinstance(student_data, dict):
        return jsonify({
            "error": "Request body must be a JSON object containing student features."
        }), 400

    try:
        predicted_score = predict_exam_score(student_data)
        return jsonify({
            "predicted_exam_score": predicted_score,
            "message": "Prediction generated successfully."
        })
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Unexpected prediction error")
        return jsonify({"error": "An unexpected error occurred while making the prediction."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
