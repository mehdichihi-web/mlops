
# ===== 1. Importer les bibliothèques =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")


# ===== 3. Charger le modèle champion au démarrage =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Décrire et valider les données reçues =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Route /health : vérifier que l'API fonctionne =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Route /predict : prédire le churn =====
@app.post("/predict")
def predict(client: Client):

    # a) Transformer le client en DataFrame d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer les variables catégorielles en colonnes indicatrices
    ligne = pd.get_dummies(ligne, columns=["abonnement"], dtype=int)

    # c) Aligner les colonnes sur celles attendues par le modèle
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Effectuer la prédiction
    resultat = int(model.predict(ligne)[0])

    # e) Retourner le résultat au format JSON
    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
