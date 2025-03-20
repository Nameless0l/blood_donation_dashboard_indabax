from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np
import os
import streamlit as st

@st.cache_resource
def train_eligibility_model(df):
    """
    Entraîne un modèle de prédiction d'éligibilité au don de sang.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données d'entraînement
        
    Returns:
        sklearn.pipeline.Pipeline: Pipeline contenant le préprocesseur et le classificateur
    """
    # Vérifier que les colonnes nécessaires existent
    required_columns = ['age', 'sexe', 'profession', 'condition_sante', 'eligible']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne {col} est nécessaire pour l'entraînement du modèle mais n'existe pas dans les données")
    
    # Sélectionner les caractéristiques et la cible
    X = df[['age', 'sexe', 'profession', 'condition_sante']]
    y = df['eligible']
    
    # Séparer en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Définir les transformateurs pour les différents types de colonnes
    numeric_features = ['age']
    categorical_features = ['sexe', 'profession', 'condition_sante']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Créer un pipeline avec prétraitement et classification
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Entraîner le modèle
    pipeline.fit(X_train, y_train)
    
    # Évaluer le modèle
    accuracy = pipeline.score(X_test, y_test)
    st.write(f"Précision du modèle: {accuracy:.4f}")
    
    # Sauvegarder le modèle
    os.makedirs('model', exist_ok=True)
    model_path = 'model/eligibility_model.pkl'
    joblib.dump(pipeline, model_path)
    
    return pipeline

def predict_eligibility(model, input_data):
    """
    Prédit l'éligibilité d'un donneur potentiel.
    
    Args:
        model: Modèle entraîné
        input_data (dict): Dictionnaire des caractéristiques du donneur
        
    Returns:
        dict: Résultat de la prédiction
    """
    # Convertir les données d'entrée en DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Faire la prédiction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "eligible": bool(prediction),
        "probability": float(probability),
        "message": "Éligible au don de sang" if prediction else "Non éligible au don de sang"
    }

def load_model():
    """
    Charge le modèle entraîné s'il existe, sinon retourne None.
    
    Returns:
        sklearn.pipeline.Pipeline or None: Le modèle chargé ou None
    """
    model_path = 'model/eligibility_model.pkl'
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None
