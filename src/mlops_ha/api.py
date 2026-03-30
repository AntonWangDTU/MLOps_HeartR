from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()

# Load your trained LogisticRegression model
model_path = Path("models/model.pkl")
model = joblib.load(model_path)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="src/mlops_ha/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Serve the HTML form with the ASCII heart.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
):
    """
    Predict heart disease using LogisticRegression.
    """
    # Convert form data to the 7-feature input array
    X = np.array([[age, sex, cp, trestbps, chol, thalach, exang]])

    # Make prediction
    pred_class = int(model.predict(X)[0])
    pred_prob = float(model.predict_proba(X)[0][1])

    return {"prediction": pred_class, "probability": pred_prob}
