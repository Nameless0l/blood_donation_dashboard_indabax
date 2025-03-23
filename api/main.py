# main.py - Fichier principal de l'API
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
import joblib
import os
import pandas as pd
import numpy as np
from enum import Enum
from pydantic import BaseModel

# Initialisation de l'API
app = FastAPI(
    title="API de prédiction d'éligibilité au don de sang",
    description="""
    Cette API permet de prédire l'éligibilité d'un donneur au don de sang en utilisant un modèle de machine learning.
    
    ## Fonctionnalités
    
    * **Prédiction d'éligibilité** - Évalue si un donneur est éligible au don de sang
    * **Détection des facteurs d'exclusion** - Identifie les raisons d'inéligibilité
    * **Niveau de confiance** - Fournit un pourcentage de confiance pour chaque prédiction
    
    ## Comment utiliser l'API
    
    1. Envoyez les données du donneur au endpoint `/predict`
    2. Recevez la prédiction d'éligibilité et les détails associés
    """,
    contact={
        "name": "Équipe médicale",
        "email": "contact@example.com",
    },
    license_info={
        "name": "Licence privée",
    },
    openapi_tags=[
        {
            "name": "Statut",
            "description": "Vérification du statut de l'API",
        },
        {
            "name": "Prédiction",
            "description": "Prédiction d'éligibilité au don de sang",
        },
        {
            "name": "Informations",
            "description": "Informations sur le modèle et ses caractéristiques",
        },
    ],
    docs_url=None,  # On désactive les routes par défaut
    redoc_url=None,
    version="1.0.0",
)

# Configuration CORS pour permettre les requêtes depuis d'autres domaines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacer par les domaines spécifiques en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemins vers les fichiers de modèle et statistiques
MODEL_PATH = "../model/eligibility_model_gradient_boosting_20250323_104955.pkl"
MODEL_INFO_PATH = "../model/model_info_20250323_104955.json"

# Classes pour les entrées et sorties
class Genre(str, Enum):
    HOMME = "Homme"
    FEMME = "Femme"

class NiveauEtude(str, Enum):
    NON_PRECISE = "Non précisé"
    PRIMAIRE = "Primaire"
    SECONDAIRE = "Secondaire"
    UNIVERSITAIRE = "Universitaire"

class SituationMatrimoniale(str, Enum):
    NON_PRECISE = "Non précisé"
    CELIBATAIRE = "Célibataire"
    MARIE = "Marié(e)"
    DIVORCE = "Divorcé(e)"
    VEUF = "Veuf/Veuve"

class Religion(str, Enum):
    NON_PRECISE = "Non précisé"
    CHRETIEN = "Chrétien(ne)"
    MUSULMAN = "Musulman(e)"
    AUTRE = "Autre"

class DejaFaitDon(str, Enum):
    OUI = "Oui"
    NON = "Non"

class DonneurInput(BaseModel):
    # Caractéristiques démographiques
    age: int = Field(..., ge=18, le=70, description="Âge du donneur (entre 18 et 70 ans)")
    genre: Genre = Field(..., description="Genre du donneur")
    niveau_etude: Optional[NiveauEtude] = Field(NiveauEtude.NON_PRECISE, description="Niveau d'études du donneur")
    situation_matrimoniale: Optional[SituationMatrimoniale] = Field(SituationMatrimoniale.NON_PRECISE, description="Situation matrimoniale du donneur")
    profession: Optional[str] = Field("Non précisé", description="Profession du donneur")
    nationalite: Optional[str] = Field("Camerounaise", description="Nationalité du donneur")
    religion: Optional[Religion] = Field(Religion.NON_PRECISE, description="Religion du donneur")
    
    # Expérience de don
    deja_donne: DejaFaitDon = Field(..., description="A déjà donné du sang")
    
    # Localisation
    arrondissement: Optional[str] = Field("Douala (Non précisé)", description="Arrondissement de résidence")
    quartier: Optional[str] = Field("Non précisé", description="Quartier de résidence")
    
    # Conditions médicales
    porteur_vih_hbs_hcv: bool = Field(False, description="Porteur de VIH, hépatite B ou C")
    diabetique: bool = Field(False, description="Diabétique")
    hypertendu: bool = Field(False, description="Hypertendu")
    asthmatique: bool = Field(False, description="Asthmatique")
    drepanocytaire: bool = Field(False, description="Drépanocytaire")
    cardiaque: bool = Field(False, description="Problèmes cardiaques")
    
    # Autres caractéristiques médicales
    taux_hemoglobine: float = Field(..., ge=7.0, le=20.0, description="Taux d'hémoglobine en g/dL")
    transfusion: bool = Field(False, description="Antécédent de transfusion")
    tatoue: bool = Field(False, description="Tatoué")
    scarifie: bool = Field(False, description="Scarifié")
    
    class Config:
        schema_extra = {
            "examples": {
                "donneur_eligible": {
                    "summary": "Donneur éligible typique",
                    "description": "Un donneur sans contre-indications médicales",
                    "value": {
                        "age": 35,
                        "genre": "Homme",
                        "niveau_etude": "Universitaire",
                        "situation_matrimoniale": "Marié(e)",
                        "profession": "Enseignant",
                        "nationalite": "Camerounaise",
                        "religion": "Chrétien(ne)",
                        "deja_donne": "Oui",
                        "arrondissement": "Douala 3",
                        "quartier": "Logbaba",
                        "porteur_vih_hbs_hcv": False,
                        "diabetique": False,
                        "hypertendu": False,
                        "asthmatique": False,
                        "drepanocytaire": False,
                        "cardiaque": False,
                        "taux_hemoglobine": 14.5,
                        "transfusion": False,
                        "tatoue": False,
                        "scarifie": False
                    }
                },
                "donneur_ineligible_1": {
                    "summary": "Donneur avec hépatite",
                    "description": "Un donneur avec une contre-indication médicale absolue",
                    "value": {
                        "age": 40,
                        "genre": "Homme",
                        "niveau_etude": "Universitaire",
                        "situation_matrimoniale": "Marié(e)",
                        "profession": "Ingénieur",
                        "nationalite": "Camerounaise",
                        "religion": "Chrétien(ne)",
                        "deja_donne": "Non",
                        "arrondissement": "Douala 5",
                        "quartier": "Kotto",
                        "porteur_vih_hbs_hcv": True,
                        "diabetique": False,
                        "hypertendu": False,
                        "asthmatique": False,
                        "drepanocytaire": False,
                        "cardiaque": False,
                        "taux_hemoglobine": 15.0,
                        "transfusion": False,
                        "tatoue": False,
                        "scarifie": False
                    }
                },
                "donneur_ineligible_2": {
                    "summary": "Donneur avec hémoglobine basse",
                    "description": "Un donneur avec un taux d'hémoglobine insuffisant",
                    "value": {
                        "age": 25,
                        "genre": "Femme",
                        "niveau_etude": "Universitaire",
                        "situation_matrimoniale": "Célibataire",
                        "profession": "Étudiante",
                        "nationalite": "Camerounaise",
                        "religion": "Chrétien(ne)",
                        "deja_donne": "Non",
                        "arrondissement": "Douala 4",
                        "quartier": "Bonabéri",
                        "porteur_vih_hbs_hcv": False,
                        "diabetique": False,
                        "hypertendu": False,
                        "asthmatique": False,
                        "drepanocytaire": False,
                        "cardiaque": False,
                        "taux_hemoglobine": 11.5,
                        "transfusion": False,
                        "tatoue": False,
                        "scarifie": False
                    }
                }
            }
        }
        

class PredictionOutput(BaseModel):
    prediction: str = Field(..., description="Prédiction d'éligibilité (Éligible ou Non éligible)")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Niveau de confiance en pourcentage")
    facteurs_importants: List[str] = Field([], description="Facteurs importants qui ont influencé la prédiction")
    raison_ineligibilite: Optional[str] = Field(None, description="Raison principale d'inéligibilité si applicable")

# Variables globales pour stocker le modèle et les caractéristiques attendues
model = None
required_columns = None
feature_stats = {}

# Fonction pour charger le modèle au démarrage
def load_model():
    global model, required_columns, feature_stats
    
    try:
        # Charger le modèle
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Modèle chargé depuis: {MODEL_PATH}")
            
            # Charger les informations du modèle si disponibles
            if os.path.exists(MODEL_INFO_PATH):
                import json
                with open(MODEL_INFO_PATH, 'r') as f:
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
            
            # Initialiser les statistiques vides pour l'instant
            # Dans une version plus complète, on pourrait charger des statistiques précalculées
            
            return True
        else:
            print(f"Modèle non trouvé à: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return False

# Fonction de prédiction avec règles de sécurité strictes
def predict_eligibility(input_data: Dict[str, Any]) -> Dict[str, Any]:
    global model, required_columns
    
    # Si le modèle n'est pas chargé, essayer de le charger
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    # Vérifier les critères d'exclusion absolus AVANT d'utiliser le modèle
    facteurs_importants = []
    raison_ineligibilite = None
    
    # Critères d'exclusion absolus
    if input_data.get('porteur_vih_hbs_hcv', False):
        facteurs_importants.append("Porteur de VIH, hépatite B ou C")
        raison_ineligibilite = "Porteur de VIH, hépatite B ou C"
        return {
            "prediction": "Non éligible",
            "confidence": 100.0,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
    
    # Autres critères d'exclusion absolus
    if input_data.get('drepanocytaire', False):
        facteurs_importants.append("Drépanocytaire")
        raison_ineligibilite = "Drépanocytaire"
        return {
            "prediction": "Non éligible",
            "confidence": 100.0,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
    
    if input_data.get('cardiaque', False):
        facteurs_importants.append("Problèmes cardiaques")
        raison_ineligibilite = "Problèmes cardiaques"
        return {
            "prediction": "Non éligible",
            "confidence": 100.0,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
    
    # Vérifier le taux d'hémoglobine
    genre = input_data.get('genre', '')
    taux_hemoglobine = input_data.get('taux_hemoglobine', 0)
    
    if (genre == "Homme" and taux_hemoglobine < 13.0) or (genre == "Femme" and taux_hemoglobine < 12.0):
        facteurs_importants.append("Taux d'hémoglobine insuffisant")
        raison_ineligibilite = "Taux d'hémoglobine insuffisant"
        return {
            "prediction": "Non éligible",
            "confidence": 95.0,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
    
    try:
        # Préparer les données pour le modèle
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
        
        # Conditions médicales (déjà vérifiées plus haut pour les critères d'exclusion)
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
        
        # Faire la prédiction
        prediction = model.predict(prediction_df)[0]
        probabilities = model.predict_proba(prediction_df)[0]
        
        # Interpréter les résultats
        if prediction == 1:
            result = "Éligible"
            confidence = probabilities[1] * 100
        else:
            result = "Non éligible"
            confidence = probabilities[0] * 100
            
            # Collecter les facteurs importants
            if input_data.get('diabetique', False):
                facteurs_importants.append("Diabète")
            if input_data.get('hypertendu', False):
                facteurs_importants.append("Hypertension")
            if input_data.get('asthmatique', False):
                facteurs_importants.append("Asthme")
            
            # Déterminer la raison principale d'inéligibilité
            if facteurs_importants:
                raison_ineligibilite = facteurs_importants[0]
        
        # VÉRIFICATION FINALE DES RÈGLES DE SÉCURITÉ
        # Même si le modèle prédit "Éligible", double-vérifier les critères d'exclusion
        if result == "Éligible":
            # Vérifier à nouveau les critères d'exclusion
            if input_data.get('porteur_vih_hbs_hcv', False):
                result = "Non éligible"
                confidence = 100.0
                facteurs_importants = ["Porteur de VIH, hépatite B ou C"]
                raison_ineligibilite = "Porteur de VIH, hépatite B ou C"
            
            elif input_data.get('drepanocytaire', False) or input_data.get('cardiaque', False):
                result = "Non éligible"
                confidence = 100.0
                if input_data.get('drepanocytaire', False):
                    facteurs_importants = ["Drépanocytaire"]
                    raison_ineligibilite = "Drépanocytaire"
                else:
                    facteurs_importants = ["Problèmes cardiaques"]
                    raison_ineligibilite = "Problèmes cardiaques"
            
            elif (genre == "Homme" and taux_hemoglobine < 13.0) or (genre == "Femme" and taux_hemoglobine < 12.0):
                result = "Non éligible"
                confidence = 95.0
                facteurs_importants = ["Taux d'hémoglobine insuffisant"]
                raison_ineligibilite = "Taux d'hémoglobine insuffisant"
        
        return {
            "prediction": result,
            "confidence": confidence,
            "facteurs_importants": facteurs_importants,
            "raison_ineligibilite": raison_ineligibilite
        }
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Route pour vérifier si l'API est en ligne
@app.get("/", tags=["Statut"])
async def root():
    return {"status": "API en ligne", "model_loaded": model is not None}

# Route pour la prédiction d'éligibilité
@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
async def predict(donneur: DonneurInput):
    # Convertir le modèle Pydantic en dictionnaire
    input_data = donneur.dict()
    
    # Faire la prédiction
    result = predict_eligibility(input_data)
    
    return PredictionOutput(
        prediction=result["prediction"],
        confidence=result["confidence"],
        facteurs_importants=result["facteurs_importants"],
        raison_ineligibilite=result["raison_ineligibilite"]
    )

# Route pour obtenir la liste des caractéristiques attendues par le modèle
@app.get("/features", tags=["Informations"])
async def get_features():
    # Si le modèle n'est pas chargé, essayer de le charger
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    return {"features": required_columns}

# Route pour obtenir des informations sur le modèle
@app.get("/model-info", tags=["Informations"])
async def get_model_info():
    # Si le modèle n'est pas chargé, essayer de le charger
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    # Charger les informations du modèle si disponibles
    if os.path.exists(MODEL_INFO_PATH):
        import json
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        return model_info
    else:
        return {
            "model_name": "gradient_boosting",
            "version": "1.0.0",
            "features": required_columns
        }

# Chargement du modèle au démarrage de l'application
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Documentation API",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1.3/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1.3/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Documentation ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# Fonction custom pour personnaliser OpenAPI (optionnel)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Cette ligne est cruciale pour corriger l'erreur :
    openapi_schema["openapi"] = "3.0.2"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Lancement de l'application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)