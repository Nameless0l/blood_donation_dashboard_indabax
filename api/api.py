from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys

# Ajout du chemin pour l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir le modèle de données pour la requête
class DonorData(BaseModel):
    age: int
    sexe: str
    profession: str
    condition_sante: str

# Créer l'application FastAPI
app = FastAPI(
    title="API de prédiction d'éligibilité au don de sang",
    description="Prédit si une personne est éligible au don de sang en fonction de ses caractéristiques",
    version="1.0.0"
)

# Charger le modèle entraîné
@app.on_event("startup")
async def startup_event():
    global model
    model_path = '../model/eligibility_model.pkl'
    try:
        model = joblib.load(model_path)
    except:
        # Si le modèle n'existe pas, on le créera à la demande
        model = None

@app.post("/predict/", response_model=dict)
async def predict_eligibility(data: DonorData):
    """
    Endpoint pour prédire l'éligibilité au don de sang.
    
    Args:
        data (DonorData): Données du donneur potentiel
        
    Returns:
        dict: Résultat de la prédiction avec l'éligibilité et la probabilité
    """
    global model
    
    try:
        # Vérifier si le modèle est chargé
        if model is None:
            # Si le modèle n'est pas chargé, on peut le créer à la demande
            # ou renvoyer une erreur
            raise HTTPException(
                status_code=503, 
                detail="Le modèle n'est pas encore disponible. Veuillez d'abord exécuter l'entraînement dans le tableau de bord."
            )
        
        # Convertir les données en DataFrame
        input_df = pd.DataFrame([{
            'age': data.age,
            'sexe': data.sexe,
            'profession': data.profession,
            'condition_sante': data.condition_sante
        }])
        
        # Faire la prédiction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return {
            "eligible": bool(prediction),
            "probability": float(probability),
            "message": "Éligible au don de sang" if prediction else "Non éligible au don de sang"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """
    Endpoint de vérification de santé de l'API.
    
    Returns:
        dict: Statut de l'API
    """
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
