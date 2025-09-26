from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError
import pandas as pd
import joblib

# Charger le modèle enregistré
model_enregistre = joblib.load('random_forest_model.pkl')

# Définition du modèle de validation des données d'entrée avec Pydantic
class DonneesEntree(BaseModel):
    Pregnancies: float  
    Glucose: float  
    BloodPressure: float 
    SkinThickness: float  
    Insulin: float  
    BMI: float  
    DiabetesPedigreeFunction: float  
    Age: float  

# Création de l'instance de l'application Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def accueil():
    """Endpoint racine qui fournit un message de bienvenue."""
    return jsonify({"message": "Bienvenue sur l'API de prédiction de diabète"})

@app.route("/predire", methods=["POST"])
def predire():
    """Endpoint pour les prédictions en utilisant le modèle chargé."""
    if not request.json:
        return jsonify({"erreur": "Aucun JSON fourni"}), 400

    try:
        # Validation des données via Pydantic
        donnees = DonneesEntree(**request.json)

        # Conversion en DataFrame
        donnees_df = pd.DataFrame([donnees.dict()])

        # Prédiction avec le modèle
        predictions = model_enregistre.predict(donnees_df)
        probabilities = model_enregistre.predict_proba(donnees_df)[:, 1]

        # Retour simplifié : uniquement prédiction et probabilité
        return jsonify({
            "prediction": int(predictions[0]),
            "probabilite_diabete": round(float(probabilities[0]), 2)
        })

    except ValidationError as ve:
        return jsonify({"erreur": ve.errors()}), 400
    except Exception as e:
        return jsonify({"erreur": str(e)}), 400

# Lancement de l'application Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)