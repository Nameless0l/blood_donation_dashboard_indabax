import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

@st.cache_data
def load_and_process_data(filepath):
    """
    Charge les données prétraitées ou tente de charger le fichier original.
    Version simplifiée qui privilégie le fichier prétraité.
    
    Args:
        filepath (str): Chemin vers le fichier de données (Excel ou CSV)
        
    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    try:
        # Vérifier si un fichier prétraité existe
        preprocessed_file = "data/preprocessed_data.csv"
        
        if os.path.exists(preprocessed_file):
            st.success(f"Utilisation du fichier prétraité: {preprocessed_file}")
            df = pd.read_csv(preprocessed_file)
            return df
        
        # Si le fichier prétraité n'existe pas, essayer de prétraiter maintenant
        st.info("Fichier prétraité non trouvé. Prétraitement en cours...")
        
        # Importer la fonction de prétraitement et l'exécuter
        from data_preprocessing import preprocess_excel_file
        preprocess_excel_file()
        
        # Vérifier si le prétraitement a créé le fichier
        if os.path.exists(preprocessed_file):
            st.success("Prétraitement réussi. Chargement des données prétraitées.")
            df = pd.read_csv(preprocessed_file)
            return df
        
        # Si toujours pas de fichier prétraité, essayer de charger le fichier original
        st.warning("Échec du prétraitement. Tentative de chargement direct...")
        
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            # Tenter de charger la feuille appropriée
            sheet_name = 'Candidat au don 2019  (avec age'
            try:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
            except:
                # Si la feuille spécifique n'existe pas, charger la première feuille
                df = pd.read_excel(filepath)
        else:
            # Charger comme CSV
            df = pd.read_csv(filepath)
            
        # Appliquer les transformations minimales nécessaires
        if 'ID' in df.columns:
            df = df.rename(columns={'ID': 'id_donneur'})
        if 'Age' in df.columns:
            df['age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(30).astype(int)
        
        # Créer une colonne d'éligibilité par défaut
        df['eligible'] = 1
        df['condition_sante'] = 'Aucune'
        
        return df
            
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        # Générer des données factices en cas d'échec
        return generate_dummy_data(500)

def prepare_geo_data(df):
    """
    Prépare les données géographiques pour la visualisation.
    
    Args:
        df (pd.DataFrame): DataFrame des données de don de sang
        
    Returns:
        pd.DataFrame: DataFrame avec données géographiques agrégées
    """
    # Vérifier que les colonnes nécessaires existent
    if 'arrondissement' not in df.columns or 'quartier' not in df.columns:
        cols = [col for col in df.columns 
                if 'arrondissement' in col.lower().replace('é', 'e') 
                or 'quartier' in col.lower().replace('é', 'e')]
        
        if len(cols) >= 2:
            # Renommer temporairement pour l'agrégation
            temp_df = df.copy()
            arrond_col = [col for col in cols if 'arrondissement' in col.lower().replace('é', 'e')][0]
            quartier_col = [col for col in cols if 'quartier' in col.lower().replace('é', 'e')][0]
            temp_df = temp_df.rename(columns={arrond_col: 'arrondissement', quartier_col: 'quartier'})
            
            # Grouper par arrondissement et quartier
            geo_data = temp_df.groupby(['arrondissement', 'quartier']).agg({
                'id_donneur': 'count',
                'eligible': 'mean'
            }).reset_index()
        else:
            # Si les colonnes n'existent vraiment pas, créer des données de démo
            st.warning("Colonnes géographiques non trouvées. Utilisation de données factices pour la démonstration.")
            
            # Créer des données factices pour la démo
            arrondissements = ["Douala 1", "Douala 2", "Douala 3", "Douala 4", "Douala 5"]
            quartiers = ["Logbaba", "Bepanda", "Bonanjo", "Deido", "Akwa", "PK 12", "Ndogbong"]
            
            data = []
            for arrond in arrondissements:
                for _ in range(3):  # 3 quartiers par arrondissement
                    quartier = np.random.choice(quartiers)
                    nb_donneurs = np.random.randint(20, 100)
                    taux_elig = np.random.uniform(0.5, 0.9)
                    data.append([arrond, quartier, nb_donneurs, taux_elig])
            
            geo_data = pd.DataFrame(data, columns=['arrondissement', 'quartier', 'nombre_donneurs', 'taux_eligibilite'])
            return geo_data
    else:
        # Si les colonnes existent, procéder normalement
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
    Adapté au format du dataset de don de sang à Douala.
    
    Args:
        n_samples (int, optional): Nombre d'échantillons à générer. Defaults to 1000.
        
    Returns:
        pd.DataFrame: DataFrame avec des données fictives
    """
    # Listes pour la génération de données aléatoires spécifiques à Douala
    arrondissements = ["Douala 1", "Douala 2", "Douala 3", "Douala 4", "Douala 5"]
    quartiers = ["Logbaba", "Bepanda", "Bonanjo", "Deido", "Akwa", "PK 12", "Ndogbong", 
                "Bonaberi", "Makepe", "Bonamoussadi", "New Bell", "Bonapriso"]
    professions = ["Étudiant(e)", "Enseignant", "Médecin", "Infirmier", "Commerçant", "Fonctionnaire", 
                  "Ouvrier", "Cadre", "Informaticien", "Militaire", "Chauffeur", "Comptable"]
    conditions_sante = ["Aucune", "Porteur HIV/HBS/HCV", "Diabétique", "Hypertendu", "Asthmatique", 
                        "Cardiaque", "Drépanocytaire"]
    sexes = ["Homme", "Femme"]
    niveaux_etude = ["Universitaire", "Secondaire", "Primaire", "Pas Précisé"]
    situations_matrimoniales = ["Célibataire", "Marié(e)", "Divorcé(e)"]
    
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