from fastapi import FastAPI, Request
import pickle
import numpy as np
import pandas as pd

app = FastAPI()


model = None
try:
    with open('../artifacts/model.pkl', 'rb') as handle:
        model = pickle.load(handle)
except Exception as e:
    model = None

@app.get("/test_model")
def test_model_loading():
    if model:
        return {"message": "Le modèle a été chargé avec succès"}
    else:
        return {"message": "Erreur lors du chargement du modèle"}

@app.get("/")
def hello_world():
    return {"message": "Hello World - FastAPI"}

@app.post("/predict")
async def predict(request: Request):
    # Récupérer les données JSON
    data = await request.json()

    # Récupération des entrées utilisateur
    input1 = data.get("ID")
    input2 = float(data.get("Amount", 0))
    input3 = float(data.get("Interest", 0))

    # Intérêt calcul
    interest_amount = input2 * (input3 / 100)

    # Lecture CSV
    df = pd.read_csv("../data/raw_data.csv", delimiter=";")

    # Mise à jour des colonnes
    df["Amount"] = input2
    df["Interest"] = interest_amount

    # Data prep
    excluded_columns = ["ID","Price"]
    training_features = [col for col in df.columns if col not in excluded_columns]
    X = df[training_features].iloc[0:1].values

    # Prediction
    prediction = model.predict_proba(X)[0, 1]
    prediction_percentage = f"{round(prediction * 100, 2)}%"

    # Résultat
    return {
        "ID": input1,
        "Amount": input2,
        "Interest": input3,
        "Prediction": prediction_percentage
    }
