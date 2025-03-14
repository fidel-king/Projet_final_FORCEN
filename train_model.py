import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Charger les données 
df = pd.read_csv("Projet 2 dataset Student_Performance.csv")

# Encodage de la variable catégorique
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# Définition des variables X et y
X = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

# Division des données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Sauvegarde du modèle
pickle.dump(model, open("student_performance_model.pkl", "wb"))

print("✅ Modèle entraîné et enregistré sous 'student_performance_model.pkl'")
