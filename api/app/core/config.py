import os
from typing import List, Dict, Any

class Settings:
    # Informations de l'application
    APP_TITLE = "API de prédiction d'éligibilité au don de sang"
    APP_DESCRIPTION = """
    Cette API permet de prédire l'éligibilité d'un donneur au don de sang en utilisant un modèle de machine learning.
    
    ## Fonctionnalités
    
    * **Prédiction d'éligibilité** - Évalue si un donneur est éligible au don de sang
    * **Détection des facteurs d'exclusion** - Identifie les raisons d'inéligibilité
    * **Niveau de confiance** - Fournit un pourcentage de confiance pour chaque prédiction
    
    ## Comment utiliser l'API
    
    1. Envoyez les données du donneur au endpoint `/predict`
    2. Recevez la prédiction d'éligibilité et les détails associés
    """
    APP_VERSION = "1.0.0"
    
    # Informations de contact
    CONTACT = {
        "name": "Équipe médicale",
        "email": "contact@example.com",
    }
    
    # Informations de licence
    LICENSE_INFO = {
        "name": "Licence privée",
    }
    
    # Chemins des fichiers de modèle
    MODEL_PATH = "./model/eligibility_model_gradient_boosting_20250323_104955.pkl"
    MODEL_INFO_PATH = "./model/model_info_20250323_104955.json"
    
    # Configuration CORS
    CORS_ORIGINS = ["*"]  # Remplacer par les domaines spécifiques en production
    
    # Tags pour OpenAPI
    OPENAPI_TAGS = [
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
    ]
    
    # Seuils pour les taux d'hémoglobine
    HEMOGLOBIN_THRESHOLD_MALE = 13.0
    HEMOGLOBIN_THRESHOLD_FEMALE = 12.0
    
    # Configuration du serveur
    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 8000))

settings = Settings()