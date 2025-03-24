import os
import joblib
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from fastapi import HTTPException

from app.core.config import settings

# Variables globales pour stocker le modèle et les caractéristiques attendues
model = None
required_columns = None
feature_stats = {}

def load_model() -> bool:
    """
    Charge le modèle de prédiction et ses métadonnées.
    
    Returns:
        bool: True si le chargement a réussi, False sinon
    """
    global model, required_columns, feature_stats
    
    try:
        # Charger le modèle
        if os.path.exists(settings.MODEL_PATH):
            model = joblib.load(settings.MODEL_PATH)
            print(f"Modèle chargé depuis: {settings.MODEL_PATH}")
            
            # Charger les informations du modèle si disponibles
            if os.path.exists(settings.MODEL_INFO_PATH):
                with open(settings.MODEL_INFO_PATH, 'r') as f:
                    model_info = json.load(f)
                
                # Extraire les caractéristiques attendues
                if 'features' in model_info:
                    required_columns = model_info['features']
                    print(f"Caractéristiques requises: {required_columns}")
            else:
                # Définir manuellement les caractéristiques si le fichier d'info n'existe pas
                required_columns = [
                    "age",
                    "experience_don",
                    "Niveau d'etude",
                    "Genre",
                    "Situation Matrimoniale (SM)",
                    "Profession",
                    "Arrondissement de résidence",
                    "Quartier de Résidence",
                    "Nationalité",
                    "Religion",
                    "A-t-il (elle) déjà donné le sang",
                    "Taux d'hémoglobine",
                    "groupe_age",
                    "arrondissement_clean",
                    "quartier_clean"
                ]
            
            return True
        else:
            print(f"Modèle non trouvé à: {settings.MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return False

def check_absolute_exclusion(input_data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[str]]:
    """
    Vérifie les critères d'exclusion absolus.
    
    Args:
        input_data: Données du donneur à évaluer
        
    Returns:
        Tuple contenant:
        - Un booléen indiquant si le donneur est exclu
        - Une liste des facteurs d'exclusion
        - La raison principale d'exclusion (si applicable)
    """
    exclusion = False
    facteurs_importants = []
    raison_ineligibilite = None
    
    # Critères d'exclusion absolus
    if input_data.get('porteur_vih_hbs_hcv', False):
        exclusion = True
        facteurs_importants.append("Porteur de VIH, hépatite B ou C")
        raison_ineligibilite = "Porteur de VIH, hépatite B ou C"
    
    elif input_data.get('drepanocytaire', False):
        exclusion = True
        facteurs_importants.append("Drépanocytaire")
        raison_ineligibilite = "Drépanocytaire"
    
    elif input_data.get('cardiaque', False):
        exclusion = True
        facteurs_importants.append("Problèmes cardiaques")
        raison_ineligibilite = "Problèmes cardiaques"
    
    # Vérifier le taux d'hémoglobine
    genre = input_data.get('genre', '')
    taux_hemoglobine = input_data.get('taux_hemoglobine', 0)
    
    if (genre == "Homme" and taux_hemoglobine < settings.HEMOGLOBIN_THRESHOLD_MALE) or \
       (genre == "Femme" and taux_hemoglobine < settings.HEMOGLOBIN_THRESHOLD_FEMALE):
        exclusion = True
        facteurs_importants.append("Taux d'hémoglobine insuffisant")
        raison_ineligibilite = "Taux d'hémoglobine insuffisant"
    
    return exclusion, facteurs_importants, raison_ineligibilite

def prepare_data_for_prediction(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prépare les données pour la prédiction.
    
    Args:
        input_data: Données du donneur à évaluer
        
    Returns:
        DataFrame prêt pour la prédiction
    """
    # Mapping entre les champs de l'API et les colonnes attendues par le modèle
    feature_mapping = {
        "age": "age",
        "genre": "Genre",
        "niveau_etude": "Niveau d'etude",
        "situation_matrimoniale": "Situation Matrimoniale (SM)",
        "profession": "Profession",
        "nationalite": "Nationalité",
        "religion": "Religion",
        "deja_donne": "A-t-il (elle) déjà donné le sang",
        "arrondissement": "Arrondissement de résidence",
        "quartier": "Quartier de Résidence",
        "taux_hemoglobine": "Taux d'hémoglobine"
    }
    
    # Créer un dictionnaire de données normalisées
    normalized_data = {}
    
    # Mapper les champs d'entrée aux colonnes attendues par le modèle
    for api_field, model_column in feature_mapping.items():
        if api_field in input_data:
            normalized_data[model_column] = input_data[api_field]
    
    # Ajouter les colonnes supplémentaires nécessaires
    normalized_data["experience_don"] = 1 if input_data.get('deja_donne') == "Oui" else 0
    normalized_data["arrondissement_clean"] = input_data.get('arrondissement', "Non précisé")
    normalized_data["quartier_clean"] = input_data.get('quartier', "Non précisé")
    
    # Calculer le groupe d'âge
    age = input_data.get('age', 35)
    if age < 18:
        age_group = "<18"
    elif age <= 25:
        age_group = "18-25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 55:
        age_group = "46-55"
    elif age <= 65:
        age_group = "56-65"
    else:
        age_group = ">65"
    normalized_data["groupe_age"] = age_group
    
    # Conditions médicales
    for condition in ['porteur_vih_hbs_hcv', 'diabetique', 'hypertendu', 'asthmatique', 
                      'drepanocytaire', 'cardiaque', 'transfusion', 'tatoue', 'scarifie']:
        normalized_data[condition] = 1 if input_data.get(condition, False) else 0
    
    # Créer un DataFrame avec une seule ligne
    prediction_df = pd.DataFrame([normalized_data])
    
    # Si nous avons une liste de colonnes requises, s'assurer que toutes sont présentes
    if required_columns:
        missing_columns = set(required_columns) - set(prediction_df.columns)
        for col in missing_columns:
            prediction_df[col] = "" if col in ["Niveau d'etude", "Genre", "Situation Matrimoniale (SM)",
                                          "Profession", "Arrondissement de résidence", "Quartier de Résidence",
                                          "Nationalité", "Religion", "A-t-il (elle) déjà donné le sang",
                                          "groupe_age", "arrondissement_clean", "quartier_clean"] else 0
    
    return prediction_df

def collect_factors(input_data: Dict[str, Any]) -> List[str]:
    """
    Collecte les facteurs importants susceptibles d'influer sur l'éligibilité.
    
    Args:
        input_data: Données du donneur
        
    Returns:
        Liste des facteurs importants
    """
    facteurs = []
    
    if input_data.get('diabetique', False):
        facteurs.append("Diabète")
    if input_data.get('hypertendu', False):
        facteurs.append("Hypertension")
    if input_data.get('asthmatique', False):
        facteurs.append("Asthme")
    if input_data.get('transfusion', False):
        facteurs.append("Antécédent de transfusion")
    if input_data.get('tatoue', False):
        facteurs.append("Tatoué")
    if input_data.get('scarifie', False):
        facteurs.append("Scarifié")
    
    return facteurs

def predict_eligibility(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prédit l'éligibilité d'un donneur au don de sang.
    
    Args:
        input_data: Données du donneur à évaluer
        
    Returns:
        Dictionnaire contenant la prédiction, le niveau de confiance,
        les facteurs importants et la raison d'inéligibilité si applicable
    
    Raises:
        HTTPException: Si le modèle n'est pas disponible ou en cas d'erreur
    """
    global model
    
    # Si le modèle n'est pas chargé, essayer de le charger
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    # Vérifier les critères d'exclusion absolus AVANT d'utiliser le modèle
    is_excluded, facteurs_importants, raison_ineligibilite = check_absolute_exclusion(input_data)
    
    if is_excluded:
        return {
            "prediction": "Non éligible",
            "confidence": 100.0,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
    
    try:
        # Préparer les données pour le modèle
        prediction_df = prepare_data_for_prediction(input_data)
        
        # Faire la prédiction
        prediction = model.predict(prediction_df)[0]
        probabilities = model.predict_proba(prediction_df)[0]
        
        # Interpréter les résultats
        if prediction == 1:
            result = "Éligible"
            confidence = probabilities[1] * 100
            facteurs_importants = []
            raison_ineligibilite = None
        else:
            result = "Non éligible"
            confidence = probabilities[0] * 100
                
            # Collecter les facteurs importants
            facteurs_importants = collect_factors(input_data)
            
            # Déterminer la raison principale d'inéligibilité
            if facteurs_importants:
                raison_ineligibilite = facteurs_importants[0]
        
        # VÉRIFICATION FINALE DES RÈGLES DE SÉCURITÉ - Double vérification
        is_excluded, exclusion_factors, exclusion_reason = check_absolute_exclusion(input_data)
        
        if is_excluded:
            result = "Non éligible"
            confidence = 100.0
            facteurs_importants = exclusion_factors
            raison_ineligibilite = exclusion_reason
        
        return {
            "prediction": result,
            "confidence": confidence,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

def get_model_info() -> Dict[str, Any]:
    """
    Récupère les informations sur le modèle.
    
    Returns:
        Dictionnaire contenant les informations du modèle
    
    Raises:
        HTTPException: Si le modèle n'est pas disponible
    """
    global model
    
    # Si le modèle n'est pas chargé, essayer de le charger
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    # Charger les informations du modèle si disponibles
    if os.path.exists(settings.MODEL_INFO_PATH):
        with open(settings.MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        return model_info
    else:
        return {
            "model_name": "gradient_boosting",
            "version": "1.0.0",
            "features": required_columns
        }

def get_features() -> List[str]:
    """
    Récupère la liste des caractéristiques requises par le modèle.
    
    Returns:
        Liste des caractéristiques
    
    Raises:
        HTTPException: Si le modèle n'est pas disponible
    """
    global model, required_columns
    
    # Si le modèle n'est pas chargé, essayer de le charger
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    return required_columns