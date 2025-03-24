"""
Fonctions utilitaires pour le prétraitement des données avant prédiction.
"""

def calculate_age_group(age: int) -> str:
    """
    Calcule le groupe d'âge à partir de l'âge.
    
    Args:
        age: L'âge en années
        
    Returns:
        Le groupe d'âge sous forme de chaîne
    """
    if age < 18:
        return "<18"
    elif age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 55:
        return "46-55"
    elif age <= 65:
        return "56-65"
    else:
        return ">65"