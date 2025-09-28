import pandas as pd
import requests
import pytest


# Charger les données de test
data = pd.read_csv('patient_a_tester.csv')
donne_predire = data.iloc[0: 1,:]
# Charger le modèle
model_enregistre = joblib.load('random_forest_model.pkl')


def test_predire():

    score = model_enregistre.predict_proba(donne_predire)[0]
    score = score[0]

    assert score >= 0 
    assert score <= 1
    """Test pour vérifier la route de prédiction"""