"""Flask web application for the Titanic Survival Predictor."""
from pathlib import Path
import json

from flask import Flask, render_template, request
from model import load, predict, train

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "metrics.json"

app = Flask(__name__)

# Load the existing trained pipeline. If model.pkl is missing, model.load()
# trains it from train.csv automatically.
pipeline = load()


def read_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@app.route("/", methods=["GET", "POST"])
def index():
    global pipeline
    result = None
    error = None
    probability = None

    form_data = {
        "name": "",
        "pclass": "3",
        "sex": "male",
        "age": "25",
        "sibsp": "0",
        "parch": "0",
        "fare": "7.25",
        "embarked": "S",
    }

    if request.method == "POST":
        form_data.update({key: request.form.get(key, "").strip() for key in form_data})
        try:
            if pipeline is None:
                pipeline = load()
            if pipeline is None:
                raise FileNotFoundError(
                    "train.csv was not found. Put train.csv in the same folder as app.py."
                )

            if form_data["sex"] not in {"male", "female"}:
                raise ValueError("Please select a valid sex.")
            if form_data["embarked"] not in {"S", "C", "Q"}:
                raise ValueError("Please select a valid embarkation port.")

            pclass = int(form_data["pclass"])
            sibsp = int(form_data["sibsp"])
            parch = int(form_data["parch"])
            fare = float(form_data["fare"])
            age_text = form_data["age"]

            if pclass not in {1, 2, 3}:
                raise ValueError("Passenger class must be 1, 2, or 3.")
            if sibsp < 0 or parch < 0:
                raise ValueError("Family counts cannot be negative.")
            if fare < 0:
                raise ValueError("Fare cannot be negative.")
            if age_text and not 0 <= float(age_text) <= 120:
                raise ValueError("Age must be between 0 and 120.")

            passenger = {
                "name": form_data["name"] or "Passenger",
                "pclass": pclass,
                "sex": form_data["sex"],
                "age": age_text,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare,
                "embarked": form_data["embarked"],
            }
            probability = predict(pipeline, passenger)
            result = "Survived" if probability >= 0.5 else "Did not survive"
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc) or "Please check the values and try again."
        except FileNotFoundError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=result,
        probability=probability,
        error=error,
        form_data=form_data,
        metrics=read_metrics(),
    )


@app.route("/retrain", methods=["POST"])
def retrain():
    global pipeline
    try:
        train()
        pipeline = load()
        return (
            "<h2>Model retrained successfully.</h2>"
            '<p><a href="/">Return to the predictor</a></p>'
        )
    except Exception as exc:
        app.logger.exception("Model retraining failed")
        return (
            "<h2>Model retraining failed.</h2>"
            f"<p>{str(exc)}</p><p><a href='/'>Return</a></p>",
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
