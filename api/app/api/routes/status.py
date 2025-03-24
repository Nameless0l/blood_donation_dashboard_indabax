from fastapi import APIRouter, Depends
from app.services.model_service import model

router = APIRouter(tags=["Statut"])

@router.get("/")
async def root():
    """
    Vérifie si l'API est en ligne et si le modèle est chargé.
    """
    return {
        "status": "API en ligne", 
        "model_loaded": model is not None
    }