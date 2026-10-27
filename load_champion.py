import mlflow
import mlflow.sklearn  # pour recharger un modèle scikit-learn
import pandas as pd

# Même base que le TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# 1. Charger le champion PAR SON ALIAS : on ne connaît ni la version ni le run
model = mlflow.sklearn.load_model("models:/churn-model@champion")
print("Modele charge :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))
# 2. Un nouveau client (valeurs au choix). Les colonnes d'abonnement sont
# déjà transformées en 0/1 : ici abonnement Standard
client = pd.DataFrame(
    [
        {
            "age": 35,
            "salaire": 2500,
            "anciennete": 1,
            "consommation": 10,
            "abonnement_Premium": False,
            "abonnement_Standard": True,
        }
    ]
)
# 3. Garder uniquement les colonnes connues du modèle, dans le bon ordre
client = client[list(model.feature_names_in_)]
# 4. Prédire : 1 = le client va partir, 0 = il reste
print(
    "Le client va-t-il partir ? ->",
    "OUI" if model.predict(client)[0] == 1 else "NON",
)
