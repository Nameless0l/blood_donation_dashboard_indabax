from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.services.model_service import get_features, get_model_info

router = APIRouter(tags=["Informations"])

@router.get("/features")
async def features() -> Dict[str, List[str]]:
    """
    Récupère la liste des caractéristiques requises par le modèle.
    
    Returns:
        Dictionnaire contenant la liste des caractéristiques
    """
    return {"features": get_features()}

@router.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Récupère les informations sur le modèle utilisé.
    
    Returns:
        Dictionnaire contenant les informations du modèle
    """
    return get_model_info()