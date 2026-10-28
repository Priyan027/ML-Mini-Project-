# Titanic Survival Predictor

## Files
- `app.py`: Flask web app and prediction route
- `model.py`: feature engineering, training, model loading and prediction
- `templates/index.html`: user interface and model metrics
- `train.csv`: Kaggle Titanic training data
- `model.pkl`: trained scikit-learn pipeline
- `metrics.json`: evaluation results
- `requirements.txt`: Python dependencies

## Run on Windows / VS Code
Open a terminal in this folder and run these commands separately:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser. Keep the terminal open while using the website.

If `model.pkl` is missing, `model.py` will train a model from `train.csv` when the app starts. To retrain all models manually, run `python model.py`.

The app expects the same feature pipeline as `model.py`: Pclass, Age, SibSp, Parch, Fare, FamilySize, IsAlone, Sex, Embarked and Title. Passenger title is extracted from the name.
