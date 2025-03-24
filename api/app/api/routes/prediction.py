from fastapi import APIRouter, HTTPException
from app.models.donor import DonneurInput
from app.models.prediction import PredictionOutput
from app.services.model_service import predict_eligibility

router = APIRouter(prefix="/predict", tags=["Prédiction"])

@router.post("", response_model=PredictionOutput)
async def predict(donneur: DonneurInput):
    """
    Prédit l'éligibilité d'un donneur au don de sang.
    
    Args:
        donneur: Données du donneur à évaluer
        
    Returns:
        Résultat de la prédiction avec le niveau de confiance et les facteurs importants
    """
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