import pandas as pd
import numpy as np
import streamlit as st
import random
from datetime import datetime, timedelta

def format_number(n):
    """
    Formate un nombre pour l'affichage (par exemple, 1000 -> 1 000).
    
    Args:
        n (int): Nombre à formater
        
    Returns:
        str: Nombre formaté
    """
    return f"{n:,}".replace(",", " ")

def create_kpi_card(title, value, delta=None, delta_color="normal"):
    """
    Crée une carte KPI pour l'affichage dans Streamlit.
    
    Args:
        title (str): Titre du KPI
        value (str): Valeur principale
        delta (str, optional): Variation. Defaults to None.
        delta_color (str, optional): Couleur de la variation ('normal', 'inverse', 'off'). Defaults to "normal".
        
    Returns:
        st.metric: Métrique Streamlit
    """
    return st.metric(title, value, delta, delta_color=delta_color)

def generate_dummy_data(n_samples=1000):
    """
    Génère des données fictives pour le développement et les tests.
    
    Args:
        n_samples (int, optional): Nombre d'échantillons à générer. Defaults to 1000.
        
    Returns:
        pd.DataFrame: DataFrame avec des données fictives
    """
    # Listes pour la génération de données aléatoires
    arrondissements = [f"Arrondissement {i}" for i in range(1, 11)]
    quartiers = [f"Quartier {i}" for i in range(1, 21)]
    professions = ["Étudiant", "Enseignant", "Médecin", "Ingénieur", "Commerçant", "Retraité", 
                   "Fonctionnaire", "Ouvrier", "Cadre", "Informaticien", "Infirmier"]
    conditions_sante = ["Aucune", "Hypertension", "Diabète", "Asthme", "VIH", "Anémie", 
                         "Allergie", "Hémophilie", "Hépatite", "Thyroïde"]
    sexes = ["Homme", "Femme"]
    
    # Générer les données
    data = {
        'id_donneur': list(range(1, n_samples + 1)),
        'age': np.random.randint(18, 70, size=n_samples),
        'sexe': np.random.choice(sexes, size=n_samples),
        'profession': np.random.choice(professions, size=n_samples),
        'arrondissement': np.random.choice(arrondissements, size=n_samples),
        'quartier': [random.choice([q for q in quartiers if int(q.split()[1]) % 2 == (int(a.split()[1]) % 2)]) 
                     for a in np.random.choice(arrondissements, size=n_samples)],
        'condition_sante': np.random.choice(conditions_sante, size=n_samples, p=[0.6, 0.1, 0.1, 0.05, 0.025, 0.025, 0.05, 0.025, 0.025, 0.05])
    }
    
    # Générer des dates de don aléatoires sur les 3 dernières années
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(n_samples)]
    data['date_don'] = dates
    
    # Déterminer l'éligibilité en fonction de la condition de santé
    # VIH, Anémie et Hémophilie rendent non éligible
    data['eligible'] = [0 if cond in ["VIH", "Anémie", "Hémophilie"] else 1 for cond in data['condition_sante']]
    
    # Créer des commentaires fictifs
    comments = [
        "Très bonne expérience, le personnel était accueillant et professionnel.",
        "Je reviendrai donner mon sang, c'était facile et rapide.",
        "L'attente était un peu longue mais le personnel était sympathique.",
        "Je me suis senti un peu faible après le don, mais content d'avoir aidé.",
        "Procédure trop longue et compliquée, je ne reviendrai pas.",
        "Personnel peu attentif, mauvaise organisation.",
        "Excellente organisation et équipe très compétente.",
        "Fier de pouvoir contribuer à sauver des vies.",
        "J'ai eu mal pendant le prélèvement, expérience désagréable.",
        "Bon accueil mais trop de questions sur ma vie privée.",
        None  # Pour simuler les valeurs manquantes
    ]
    
    # Attribuer des commentaires aléatoirement, avec 30% de valeurs manquantes
    data['commentaire'] = [random.choice(comments) if random.random() > 0.3 else None for _ in range(n_samples)]
    
    return pd.DataFrame(data)

def format_date(date_str):
    """
    Formate une chaîne de date en format lisible.
    
    Args:
        date_str (str): Chaîne de date au format ISO
        
    Returns:
        str: Date formatée
    """
    try:
        date_obj = pd.to_datetime(date_str)
        return date_obj.strftime('%d/%m/%Y')
    except:
        return date_str
