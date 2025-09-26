import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:5000/predire"

def envoyer_pour_prediction(donnees):
    try:
        response = requests.post(API_URL, json=donnees)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Prédiction du Diabète", layout="wide")
    st.title("Prédiction du Diabète")

    # Onglets principaux
    tab1, tab2 = st.tabs(["Prédiction individuelle", "Analyse de groupe"])

    # --------------------------------
    # Onglet 1 : Prédiction individuelle
    # --------------------------------
    with tab1:
        st.write("Saisissez les données cliniques du patient pour estimer le risque de diabète.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Données du patient")
            pregnancies = st.number_input("Nombre de grossesses", min_value=0, step=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, step=0.1)
            blood_pressure = st.number_input("Pression artérielle (mm Hg)", min_value=0.0, step=0.1)
            skin_thickness = st.number_input("Épaisseur de peau (mm)", min_value=0.0, step=0.1)
            insulin = st.number_input("Insuline (µU/mL)", min_value=0.0, step=0.1)
            bmi = st.number_input("Indice de masse corporelle (BMI)", min_value=0.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
            age = st.number_input("Âge (années)", min_value=0, step=1)

            if st.button("Lancer la prédiction"):
                donnees_patient = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age
                }

                resultat = envoyer_pour_prediction(donnees_patient)

                if resultat:
                    prob = round(resultat['probabilite_diabete'] * 100, 2)
                    color = "green" if prob < 50 else "red"

                    with col2:
                        st.subheader("Résultat de la prédiction")
                        st.metric("Prédiction", "Diabétique" if resultat['prediction']==1 else "Non diabétique")
                        st.write(f"Probabilité estimée : **{prob}%**")

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob,
                            number={'suffix': '%'},
                            gauge={
                                'axis': {'range':[0,100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range':[0,50], 'color':'lightgreen'},
                                    {'range':[50,100], 'color':'lightcoral'}
                                ]
                            },
                            domain={'x':[0,1], 'y':[0,1]}
                        ))
                        fig.update_layout(margin={'t':0,'b':0,'l':0,'r':0}, height=300)
                        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------
    # Onglet 2 : Analyse de groupe
    # --------------------------------
    with tab2:
        st.write("Analyse de groupe pour visualiser le risque et les tendances dans vos données.")

        fichier = st.file_uploader("Choisissez un fichier CSV pour l'analyse de groupe", type=["csv"])

        if fichier is not None:
            try:
                df = pd.read_csv(fichier)
                st.subheader("Aperçu des données")
                st.dataframe(df.head())

                # --- Sous-onglets après l'aperçu ---
                tab_patient, tab_moyenne, tab_hist = st.tabs([
                    "Nombre de patients diabétiques / non diabétiques",
                    "Moyenne des variables clés",
                    "Histogramme du risque par tranche d'âge"
                ])

                # ---- Sous-onglet 1 : Nombre de patients ----
                with tab_patient:
                    if "Outcome" in df.columns:
                        diabetiques = df["Outcome"].sum()
                        non_diabetiques = len(df) - diabetiques
                        st.metric("Patients diabétiques", diabetiques)
                        st.metric("Patients non diabétiques", non_diabetiques)
                        st.write(f"Total : {len(df)} patients")

                # ---- Sous-onglet 2 : Moyenne des variables clés ----
                with tab_moyenne:
                    variables_cles = ["Glucose", "BMI", "BloodPressure", "Insulin"]
                    if all(v in df.columns for v in variables_cles):
                        st.dataframe(df[variables_cles].mean().to_frame("Moyenne"))

                # ---- Sous-onglet 3 : Histogramme du risque par âge ----
                with tab_hist:
                    if "Age" in df.columns and "Outcome" in df.columns:
                        df['Tranche d\'âge'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70,80,90], 
                                                       labels=["20-30","31-40","41-50","51-60","61-70","71-80","81-90"])
                        hist = px.histogram(df, x="Tranche d'âge", color="Outcome", barmode='group',
                                            labels={"Outcome":"Diabète"})
                        st.plotly_chart(hist, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier : {str(e)}")

if __name__ == "__main__":
    main()