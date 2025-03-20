import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

@st.cache_data
def load_and_process_data(filepath):
    """
    Charge et prétraite les données des campagnes de don de sang.
    Adaptation spécifique pour le dataset du challenge.
    
    Args:
        filepath (str): Chemin vers le fichier Excel des données
        
    Returns:
        pd.DataFrame: DataFrame nettoyé et prétraité
    """
    try:
        # Vérifier si le fichier est un Excel (avec plusieurs feuilles)
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            # Charger les deux feuilles principales
            df_2019 = pd.read_excel(filepath, sheet_name='Sheet 2019')
            df_volontaire = pd.read_excel(filepath, sheet_name='Sheet Volontaire')
            
            # Fusionner les données si nécessaire ou choisir une des feuilles
            # Pour cet exemple, nous allons utiliser la feuille 'Sheet Volontaire'
            df = df_volontaire
        else:
            # Si c'est un CSV ou autre format
            df = pd.read_csv(filepath)
        
        # Renommer les colonnes pour faciliter le traitement
        column_mapping = {
            'ID': 'id_donneur',
            'Age': 'age',
            'Horodateur': 'date_don',
            'Niveau_d\'etude': 'niveau_etude',
            'Genre_': 'sexe',
            'Taille_': 'taille',
            'Poids': 'poids',
            'Situation_Matrimoniale_(SM)': 'situation_matrimoniale',
            'Profession_': 'profession',
            'Arrondissement_de_résidence_': 'arrondissement',
            'Quartier_de_Résidence_': 'quartier',
        }
        
        # Appliquer le mapping seulement pour les colonnes qui existent
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Traiter les colonnes d'éligibilité
        health_conditions = [
            'Raison_de_non-eligibilité_totale__[Porteur(HIV,hbs,hcv)]',
            'Raison_de_non-eligibilité_totale__[Opéré]',
            'Raison_de_non-eligibilité_totale__[Drepanocytaire]',
            'Raison_de_non-eligibilité_totale__[Diabétique]',
            'Raison_de_non-eligibilité_totale__[Hypertendus]',
            'Raison_de_non-eligibilité_totale__[Asthmatiques]',
            'Raison_de_non-eligibilité_totale__[Cardiaque]',
            'Raison_de_non-eligibilité_totale__[Tatoué]',
            'Raison_de_non-eligibilité_totale__[Scarifié]'
        ]
        
        # Créer des colonnes pour chaque condition de santé
        for condition in health_conditions:
            short_name = condition.split('[')[-1].replace(']', '').lower().strip()
            if condition in df.columns:
                # Convertir les réponses en booléen (1 pour Oui, 0 pour Non)
                df[short_name] = df[condition].apply(
                    lambda x: 1 if isinstance(x, str) and x.lower() in ['oui', 'yes', '1'] else 0
                )
        
        # Créer une colonne d'éligibilité
        # Un donneur est non éligible s'il a au moins une des conditions de non-éligibilité
        if 'porteur(hiv,hbs,hcv)' in df.columns:
            df['eligible'] = ~(
                (df['porteur(hiv,hbs,hcv)'] == 1) | 
                (df['drepanocytaire'] == 1) |
                (df['opéré'] == 1)  # On considère qu'une opération récente rend non éligible
            )
            df['eligible'] = df['eligible'].astype(int)
        else:
            # Si les colonnes spécifiques n'existent pas, créer une colonne factice
            df['eligible'] = np.random.choice([0, 1], size=len(df), p=[0.2, 0.8])
        
        # Traiter la colonne de date
        if 'date_don' in df.columns:
            try:
                df['date_don'] = pd.to_datetime(df['date_don'], errors='coerce')
            except:
                # Si conversion échoue, garder tel quel
                pass
        
        # Gérer les valeurs manquantes
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median())
        if 'sexe' in df.columns:
            df['sexe'] = df['sexe'].fillna(df['sexe'].mode()[0])
        if 'profession' in df.columns:
            df['profession'] = df['profession'].fillna('Non spécifié')
        if 'taille' in df.columns:
            df['taille'] = pd.to_numeric(df['taille'], errors='coerce')
            df['taille'] = df['taille'].fillna(df['taille'].median())
        if 'poids' in df.columns:
            df['poids'] = pd.to_numeric(df['poids'], errors='coerce')
            df['poids'] = df['poids'].fillna(df['poids'].median())
        
        # Créer une colonne 'condition_sante' basée sur les drapeaux de santé
        df['condition_sante'] = 'Aucune'
        if 'porteur(hiv,hbs,hcv)' in df.columns and df['porteur(hiv,hbs,hcv)'].sum() > 0:
            df.loc[df['porteur(hiv,hbs,hcv)'] == 1, 'condition_sante'] = 'Porteur HIV/HBS/HCV'
        if 'diabétique' in df.columns and df['diabétique'].sum() > 0:
            df.loc[df['diabétique'] == 1, 'condition_sante'] = 'Diabétique'
        if 'hypertendus' in df.columns and df['hypertendus'].sum() > 0:
            df.loc[df['hypertendus'] == 1, 'condition_sante'] = 'Hypertendu'
        if 'asthmatiques' in df.columns and df['asthmatiques'].sum() > 0:
            df.loc[df['asthmatiques'] == 1, 'condition_sante'] = 'Asthmatique'
        if 'cardiaque' in df.columns and df['cardiaque'].sum() > 0:
            df.loc[df['cardiaque'] == 1, 'condition_sante'] = 'Cardiaque'
        if 'drepanocytaire' in df.columns and df['drepanocytaire'].sum() > 0:
            df.loc[df['drepanocytaire'] == 1, 'condition_sante'] = 'Drépanocytaire'
        
        # Traiter la colonne de commentaires
        comment_column = 'Si_autres_raison_préciser_'
        if comment_column in df.columns:
            df = df.rename(columns={comment_column: 'commentaire'})
        
        # Vérifier si les colonnes d'arrondissement et quartier existent
        if 'arrondissement' not in df.columns:
            df['arrondissement'] = 'Non spécifié'
        if 'quartier' not in df.columns:
            df['quartier'] = 'Non spécifié'
            
        # Créer des caractéristiques dérivées
        if 'date_don' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_don']):
            df['mois_don'] = df['date_don'].dt.month
            df['annee_don'] = df['date_don'].dt.year
            df['trimestre_don'] = df['date_don'].dt.quarter
        
        # Créer des tranches d'âge si la colonne âge existe
        if 'age' in df.columns:
            bins = [18, 25, 35, 45, 55, 65, 100]
            labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            df['tranche_age'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        
        # Retourner le dataframe nettoyé et prétraité
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        # En cas d'erreur, retourner un DataFrame vide ou générer des données factices
        return generate_dummy_data(500)

def prepare_geo_data(df):
    """
    Prépare les données géographiques pour la visualisation.
    
    Args:
        df (pd.DataFrame): DataFrame des données de don de sang
        
    Returns:
        pd.DataFrame: DataFrame avec données géographiques agrégées
    """
    # Grouper par arrondissement et quartier
    geo_data = df.groupby(['arrondissement', 'quartier']).agg({
        'id_donneur': 'count',
        'eligible': 'mean'
    }).reset_index()
    
    geo_data.columns = ['arrondissement', 'quartier', 'nombre_donneurs', 'taux_eligibilite']
    geo_data['taux_eligibilite'] = (geo_data['taux_eligibilite'] * 100).round(2)
    
    return geo_data

def generate_dummy_data(n_samples=1000):
    """
    Génère des données fictives pour le développement et les tests.
    Adapté au format du dataset de don de sang.
    
    Args:
        n_samples (int, optional): Nombre d'échantillons à générer. Defaults to 1000.
        
    Returns:
        pd.DataFrame: DataFrame avec des données fictives
    """
    # Listes pour la génération de données aléatoires
    arrondissements = ["Douala 1", "Douala 2", "Douala 3", "Douala 4", "Douala 5"]
    quartiers = ["Logbaba", "Bepanda", "Bonanjo", "Deido", "Akwa", "PK 12", "Ndogbong", 
                "Bonaberi", "Makepe", "Bonamoussadi", "New Bell", "Bonapriso"]
    professions = ["Étudiant(e)", "Enseignant", "Médecin", "Ingénieur", "Commerçant", "Retraité", 
                  "Fonctionnaire", "Ouvrier", "Cadre", "Informaticien", "Infirmier", "Militaire"]
    conditions_sante = ["Aucune", "Porteur HIV/HBS/HCV", "Diabétique", "Hypertendu", "Asthmatique", 
                        "Cardiaque", "Drépanocytaire"]
    sexes = ["Homme", "Femme"]
    niveaux_etude = ["Universitaire", "Secondaire", "Primaire", "Pas Précisé", "Aucun"]
    situations_matrimoniales = ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf(ve)"]
    
    # Générer les données
    data = {
        'id_donneur': [f"DONOR_{i}" for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 70, size=n_samples),
        'sexe': np.random.choice(sexes, size=n_samples),
        'profession': np.random.choice(professions, size=n_samples),
        'arrondissement': np.random.choice(arrondissements, size=n_samples),
        'quartier': np.random.choice(quartiers, size=n_samples),
        'niveau_etude': np.random.choice(niveaux_etude, size=n_samples),
        'situation_matrimoniale': np.random.choice(situations_matrimoniales, size=n_samples),
        'taille': np.random.uniform(1.5, 2.0, size=n_samples).round(2),
        'poids': np.random.uniform(50, 100, size=n_samples).round(1)
    }
    
    # Générer les drapeaux de conditions de santé
    conditions = ['porteur(hiv,hbs,hcv)', 'diabétique', 'hypertendu', 'asthmatiques', 
                  'cardiaque', 'drepanocytaire', 'opéré', 'tatoué', 'scarifié']
    
    for condition in conditions:
        # Générer avec une faible probabilité (5-10%)
        prob = 0.05 if condition == 'porteur(hiv,hbs,hcv)' else 0.1
        data[condition] = np.random.choice([0, 1], size=n_samples, p=[1-prob, prob])
    
    # Générer des dates de don aléatoires sur les 3 dernières années
    end_date = datetime.now()
    start_date = datetime(end_date.year - 3, end_date.month, end_date.day)
    days_range = (end_date - start_date).days
    random_days = [np.random.randint(0, days_range) for _ in range(n_samples)]
    dates = [start_date + pd.Timedelta(days=days) for days in random_days]
    data['date_don'] = dates
    
    # Déterminer l'éligibilité en fonction des conditions de santé
    data['eligible'] = [
        0 if (hiv or drep) else 1 
        for hiv, drep in zip(data['porteur(hiv,hbs,hcv)'], data['drepanocytaire'])
    ]
    
    # Créer la colonne condition_sante
    df = pd.DataFrame(data)
    df['condition_sante'] = 'Aucune'
    df.loc[df['porteur(hiv,hbs,hcv)'] == 1, 'condition_sante'] = 'Porteur HIV/HBS/HCV'
    df.loc[df['diabétique'] == 1, 'condition_sante'] = 'Diabétique'
    df.loc[df['hypertendu'] == 1, 'condition_sante'] = 'Hypertendu'
    df.loc[df['asthmatiques'] == 1, 'condition_sante'] = 'Asthmatique'
    df.loc[df['cardiaque'] == 1, 'condition_sante'] = 'Cardiaque'
    df.loc[df['drepanocytaire'] == 1, 'condition_sante'] = 'Drépanocytaire'
    
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
    df['commentaire'] = [np.random.choice(comments) if np.random.random() > 0.3 else None for _ in range(n_samples)]
    
    return df