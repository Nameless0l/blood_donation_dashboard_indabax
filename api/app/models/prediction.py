from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionOutput(BaseModel):
    prediction: str = Field(..., description="Prédiction d'éligibilité (Éligible ou Non éligible)")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Niveau de confiance en pourcentage")
    facteurs_importants: List[str] = Field([], description="Facteurs importants qui ont influencé la prédiction")
    raison_ineligibilite: Optional[str] = Field(None, description="Raison principale d'inéligibilité si applicable")