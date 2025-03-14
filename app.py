import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Performance Étudiante",
    page_icon="📚",
    layout="centered"
)

# Charger le modèle de régression linéaire
with open("student_performance_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialiser une liste pour stocker les prédictions
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# Appliquer du CSS pour améliorer le design
st.markdown(
    """
    <style>
        body {
            background-color: #333333;
        }
        .stApp {
            background-color: #333333;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #f0f0f0;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Titre de l'application
st.markdown("<h1 style='text-align: center; color: #ffffff;'>📚 Prédiction de la Performance Étudiante 🎓</h1>", unsafe_allow_html=True)
st.markdown("### **Remplissez les informations suivantes pour estimer votre performance académique.**")

# Charger le modèle entraîné
model = pickle.load(open("student_performance_model.pkl", "rb"))

# Saisie des données utilisateur
hours_studied = st.slider("⏳ Heures étudiées par jour", 1, 10, 5)
previous_scores = st.slider("📊 Score précédent", 40, 100, 70)
extracurricular = st.selectbox("🎭 Activités extrascolaires", ["Oui", "Non"])
sleep_hours = st.slider("😴 Heures de sommeil", 4, 10, 7)
sample_questions = st.slider("📄 Nombre de questions papier pratiquées", 0, 10, 5)

# Encodage de la variable catégorique (Oui = 1, Non = 0)
extracurricular_encoded = 1 if extracurricular == "Oui" else 0

# Bouton de prédiction
if st.button("📈 Prédire la Performance"):
    # Préparation des données
    input_data = np.array([[hours_studied, previous_scores, extracurricular_encoded, sleep_hours, sample_questions]])
    
    # Prédiction avec le modèle de Régression Linéaire
    predicted_score = model.predict(input_data)[0]
    # Ajouter le résultat à la session pour sauvegarde
    st.session_state["predictions"].append({
        "Heures étudiées": hours_studied,
        "Score précédent": previous_scores,
        "Activités extrascolaires": "Oui" if extracurricular_encoded == 1 else "Non",
        "Heures de sommeil": sleep_hours,
        "Questions pratiquées": sample_questions,
        "Score Prédit": round(predicted_score, 2)
    })

    # Affichage du résultat
    st.success(f"🎯 **Votre score de performance estimé est : {round(predicted_score, 2)} / 100**")
    
    # Graphique de visualisation du score
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Score Prédit"], [predicted_score], color=['#4CAF50'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("📊 Visualisation de votre performance")
    st.pyplot(fig)

    # Conseils personnalisés
    st.markdown("### 📌 **Conseils pour améliorer votre performance** :")
    
    if predicted_score > 80:
        st.success("✅ **Excellent !** Continuez ainsi, vous êtes sur la bonne voie ! 🚀")
    elif predicted_score > 50:
        st.warning("⚡ **Bon travail !** Mais vous pouvez encore vous améliorer en pratiquant plus d'exercices ! 💪")
    else:
        st.error("❗ **Attention !** Vous devriez augmenter vos heures d'étude et pratiquer davantage ! 📚")

# Statistiques dynamiques
if len(st.session_state["predictions"]) > 1:
    df_predictions = pd.DataFrame(st.session_state["predictions"])
    
    st.markdown("## 📊 **Statistiques sur les prédictions récentes**")

    # Affichage de la moyenne des scores prédits
    avg_score = df_predictions["Score Prédit"].mean()
    st.info(f"📈 **Moyenne des performances estimées : {round(avg_score, 2)} / 100**")
    # Bouton pour exporter les résultats en CSV
    st.markdown("### 📥 **Exporter les résultats en CSV**")
    
    csv_file = "predictions.csv"
    df_predictions.to_csv(csv_file, index=False)
    
    with open(csv_file, "rb") as file:
        st.download_button(
            label="📥 Télécharger les résultats",
            data=file,
            file_name="predictions.csv",
            mime="text/csv"
        )

# Message de fin
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #f0f0f0;'>📌 **Un bon équilibre entre travail et repos améliore la performance académique !** 💡</h4>", unsafe_allow_html=True)