from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

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