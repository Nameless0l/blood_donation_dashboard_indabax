import pandas as pd
import numpy as np
from datetime import datetime
import re

def preprocess_data(file_path):
    """
    Fonction principale pour prétraiter les données du fichier Excel contenant
    les informations sur les campagnes de don de sang.
    
    Args:
        file_path (str): Chemin vers le fichier Excel
        
    Returns:
        dict: Dictionnaire contenant les différents DataFrames prétraités
    """
    # Charger les trois feuilles du fichier Excel
    print("Chargement des données...")
    candidats_df = pd.read_excel(file_path, sheet_name=0)
    donneurs_df = pd.read_excel(file_path, sheet_name=1)
    candidats_age_df = pd.read_excel(file_path, sheet_name=2)
    
    # Prétraiter chaque DataFrame
    candidats_clean = preprocess_candidats(candidats_df)
    donneurs_clean = preprocess_donneurs(donneurs_df)
    candidats_age_clean = preprocess_candidats_age(candidats_age_df)
    
    # Créer un DataFrame combiné pour l'analyse
    combined_df = create_combined_dataset(candidats_clean, donneurs_clean, candidats_age_clean)
    
    return {
        'candidats': candidats_clean,
        'donneurs': donneurs_clean,
        'candidats_age': candidats_age_clean,
        'combined': combined_df
    }

def preprocess_candidats(df):
    """
    Prétraite les données des candidats au don
    
    Args:
        df (DataFrame): DataFrame brut des candidats
        
    Returns:
        DataFrame: DataFrame prétraité
    """
    print("Prétraitement des données des candidats...")
    
    # Faire une copie pour éviter de modifier l'original
    clean_df = df.copy()
    
    # Nettoyer les dates
    clean_df = clean_dates(clean_df)
    
    # Standardiser les noms de colonnes
    clean_df.columns = [col.strip() for col in clean_df.columns]
    
    # Convertir l'éligibilité en valeurs numériques pour faciliter l'analyse
    eligibility_map = {
        'Eligible': 1, 
        'Temporairement Non-eligible': 0, 
        'Définitivement non-eligible': -1
    }
    
    clean_df['eligibilite_code'] = clean_df['ÉLIGIBILITÉ AU DON.'].map(eligibility_map)
    
    # Créer des indicateurs pour les conditions de santé
    health_conditions = [
        "Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]",
        "Raison de non-eligibilité totale  [Diabétique]",
        "Raison de non-eligibilité totale  [Hypertendus]",
        "Raison de non-eligibilité totale  [Asthmatiques]",
        "Raison de non-eligibilité totale  [Drepanocytaire]",
        "Raison de non-eligibilité totale  [Cardiaque]"
    ]
    
    for condition in health_conditions:
        col_name = condition.split('[')[1].split(']')[0].strip()
        clean_df[f'{col_name}_indicateur'] = clean_df[condition].apply(
            lambda x: 1 if x == 'Oui' else 0 if x == 'Non' else np.nan
        )
    
    # Nettoyer et standardiser les arrondissements
    clean_df['arrondissement_clean'] = clean_df['Arrondissement de résidence'].apply(clean_arrondissement)
    
    # Nettoyer et standardiser les quartiers
    clean_df['quartier_clean'] = clean_df['Quartier de Résidence'].apply(lambda x: str(x).strip())
    
    # Créer une variable pour l'expérience de don de sang antérieure
    clean_df['experience_don'] = clean_df['A-t-il (elle) déjà donné le sang'].apply(
        lambda x: 1 if x == 'Oui' else 0 if x == 'Non' else np.nan
    )
    
    # Calculer le temps depuis le dernier don (en jours)
    clean_df['jours_depuis_dernier_don'] = clean_df.apply(
        lambda row: calculate_days_since_donation(row['Date de remplissage de la fiche'], 
                                                 row['Si oui preciser la date du dernier don.'])
        if row['A-t-il (elle) déjà donné le sang'] == 'Oui' else np.nan, 
        axis=1
    )
    
    return clean_df

def preprocess_donneurs(df):
    """
    Prétraite les données des donneurs
    
    Args:
        df (DataFrame): DataFrame brut des donneurs
        
    Returns:
        DataFrame: DataFrame prétraité
    """
    print("Prétraitement des données des donneurs...")
    
    # Faire une copie pour éviter de modifier l'original
    clean_df = df.copy()
    
    # Standardiser les noms de colonnes
    clean_df.columns = [col.strip() for col in clean_df.columns]
    
    # Convertir 'Horodateur' en datetime
    if 'Horodateur' in clean_df.columns:
        clean_df['Horodateur'] = pd.to_datetime(clean_df['Horodateur'], errors='coerce')
        
        # Extraire date et heure
        clean_df['date_don'] = clean_df['Horodateur'].dt.date
        clean_df['mois_don'] = clean_df['Horodateur'].dt.month
        clean_df['jour_semaine_don'] = clean_df['Horodateur'].dt.dayofweek
        clean_df['annee_don'] = clean_df['Horodateur'].dt.year
    
    # Standardiser genre
    if 'Sexe' in clean_df.columns:
        gender_map = {'M': 'Homme', 'F': 'Femme'}
        clean_df['Genre'] = clean_df['Sexe'].map(gender_map)
    
    # Catégoriser les groupes d'âge
    if 'Age ' in clean_df.columns:
        # Nettoyer d'abord les âges problématiques (comme les valeurs extrêmes)
        clean_df['Age '] = clean_df['Age '].apply(lambda x: x if isinstance(x, (int, float)) and 0 < x < 120 else np.nan)
        
        # Créer des tranches d'âge
        age_bins = [0, 18, 25, 35, 45, 55, 65, 120]
        age_labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '>65']
        clean_df['groupe_age'] = pd.cut(clean_df['Age '], bins=age_bins, labels=age_labels, right=False)
    
    # Catégoriser les groupes sanguins
    if 'Groupe Sanguin ABO / Rhesus ' in clean_df.columns:
        # Extraire le groupe sanguin et le rhésus
        clean_df['groupe_sanguin'] = clean_df['Groupe Sanguin ABO / Rhesus '].str.extract(r'([ABO]+)')
        clean_df['rhesus'] = clean_df['Groupe Sanguin ABO / Rhesus '].str.extract(r'([+-])')
    
    # Type de don
    if 'Type de donation ' in clean_df.columns:
        donation_map = {'F': 'Don standard', 'B': 'Don de composant sanguin'}
        clean_df['type_donation'] = clean_df['Type de donation '].map(donation_map)
    
    return clean_df

def preprocess_candidats_age(df):
    """
    Prétraite les données des candidats avec âge
    
    Args:
        df (DataFrame): DataFrame brut des candidats avec âge
        
    Returns:
        DataFrame: DataFrame prétraité
    """
    print("Prétraitement des données des candidats avec âge...")
    
    # Faire une copie pour éviter de modifier l'original
    clean_df = df.copy()
    
    # Standardiser les noms de colonnes
    clean_df.columns = [col.strip().replace('_', ' ') for col in clean_df.columns]
    
    # Catégoriser les groupes d'âge
    if 'Age' in clean_df.columns:
        # Nettoyer d'abord les âges problématiques
        clean_df['Age'] = clean_df['Age'].apply(lambda x: x if isinstance(x, (int, float)) and 0 < x < 120 else np.nan)
        
        # Créer des tranches d'âge
        age_bins = [0, 18, 25, 35, 45, 55, 65, 120]
        age_labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '>65']
        clean_df['groupe_age'] = pd.cut(clean_df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Gérer les colonnes en convertissant les noms
    # Transformer les colonnes qui ont été renommées
    column_mapping = {
        'ÉLIGIBILITÉ AU DON.': 'ÉLIGIBILITÉ AU DON',
        'Genre ': 'Genre',
        # Ajouter d'autres mappages si nécessaire
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in clean_df.columns:
            clean_df[new_col] = clean_df[old_col]
    
    return clean_df

def create_combined_dataset(candidats_df, donneurs_df, candidats_age_df):
    """
    Crée un dataset combiné pour l'analyse
    
    Args:
        candidats_df (DataFrame): DataFrame des candidats prétraité
        donneurs_df (DataFrame): DataFrame des donneurs prétraité
        candidats_age_df (DataFrame): DataFrame des candidats avec âge prétraité
        
    Returns:
        DataFrame: DataFrame combiné pour l'analyse
    """
    print("Création d'un dataset combiné pour l'analyse...")
    
    # Créer une base avec les candidats
    combined_df = candidats_df.copy()
    
    # Ajouter des informations d'âge depuis candidats_age_df si possible
    # Ici, nous supposons qu'il n'y a pas de correspondance directe entre les DataFrames
    # Dans un cas réel, nous aurions besoin d'une clé commune pour joindre les DataFrames
    
    # Ajouter des statistiques agrégées sur les donneurs
    # Par exemple, la distribution des groupes sanguins, la saisonnalité des dons, etc.
    
    return combined_df

def clean_dates(df):
    """
    Nettoie et standardise les colonnes de dates
    
    Args:
        df (DataFrame): DataFrame contenant des dates
        
    Returns:
        DataFrame: DataFrame avec dates nettoyées
    """
    # Colonnes de dates à nettoyer
    date_columns = [
        'Date de remplissage de la fiche',
        'Date de naissance',
        'Si oui preciser la date du dernier don.',
        'Date de dernières règles (DDR) '
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Créer l'âge si 'Date de naissance' est disponible
    if 'Date de naissance' in df.columns and 'Date de remplissage de la fiche' in df.columns:
        df['age'] = df.apply(
            lambda row: calculate_age(row['Date de naissance'], row['Date de remplissage de la fiche'])
            if pd.notna(row['Date de naissance']) and pd.notna(row['Date de remplissage de la fiche']) 
            else np.nan,
            axis=1
        )
        
        # Créer des tranches d'âge
        age_bins = [0, 18, 25, 35, 45, 55, 65, 120]
        age_labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '>65']
        df['groupe_age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    return df

def calculate_age(birth_date, reference_date):
    """
    Calcule l'âge en années entre deux dates
    
    Args:
        birth_date (datetime): Date de naissance
        reference_date (datetime): Date de référence
        
    Returns:
        int: Âge en années
    """
    if pd.isna(birth_date) or pd.isna(reference_date):
        return np.nan
        
    try:
        # Gérer les formats de date incorrects
        if birth_date.year < 1900 or birth_date.year > reference_date.year:
            return np.nan
            
        age = reference_date.year - birth_date.year
        
        # Ajuster l'âge si l'anniversaire n'est pas encore passé cette année
        if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
            age -= 1
            
        return age
    except:
        return np.nan

def calculate_days_since_donation(reference_date, last_donation_date):
    """
    Calcule le nombre de jours écoulés depuis le dernier don
    
    Args:
        reference_date (datetime): Date de référence
        last_donation_date (datetime): Date du dernier don
        
    Returns:
        int: Nombre de jours
    """
    if pd.isna(reference_date) or pd.isna(last_donation_date):
        return np.nan
        
    try:
        # Vérifier que les dates sont cohérentes
        if last_donation_date > reference_date:
            return np.nan
            
        return (reference_date - last_donation_date).days
    except:
        return np.nan

def clean_arrondissement(arrond):
    """
    Nettoie et standardise les noms d'arrondissements
    
    Args:
        arrond (str): Nom d'arrondissement brut
        
    Returns:
        str: Nom d'arrondissement standardisé
    """
    if pd.isna(arrond) or arrond == "":
        return "Non précisé"
        
    arrond = str(arrond).strip()
    
    # Standardiser les arrondissements de Douala
    if "douala" in arrond.lower() and "non précisé" not in arrond.lower():
        # Extraire le numéro d'arrondissement s'il existe
        match = re.search(r'douala\s*(\d+)', arrond.lower())
        if match:
            return f"Douala {match.group(1)}"
        else:
            return "Douala (Non précisé)"
    
    return arrond

def save_processed_data(data_dict, output_folder="data/processed_data"):
    """
    Sauvegarde les données prétraitées dans des fichiers CSV
    
    Args:
        data_dict (dict): Dictionnaire contenant les DataFrames prétraités
        output_folder (str): Dossier où sauvegarder les fichiers
    """
    import os
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Sauvegarder chaque DataFrame
    for name, df in data_dict.items():
        output_path = os.path.join(output_folder, f"{name}_processed.csv")
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Fichier sauvegardé: {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    file_path = "./data/dataset.xlsx"
    
    # Prétraiter les données
    processed_data = preprocess_data(file_path)
    
    # Sauvegarder les données prétraitées
    save_processed_data(processed_data)
    
    print("Prétraitement terminé!")
    
# Ajouter cette fonction à preprocessing.py pour gérer les fichiers CSV

def preprocess_csv_data(csv_path):
    """
    Prétraite un fichier CSV avec la même structure que le dataset de don de sang
    
    Args:
        csv_path (str): Chemin vers le fichier CSV
        
    Returns:
        dict: Dictionnaire contenant les DataFrames prétraités
    """
    import pandas as pd
    import numpy as np
    
    print("Chargement des données CSV...")
    df = pd.read_csv(csv_path)
    
    # Convertir les colonnes de date si elles existent
    date_columns = [
        'Date de remplissage de la fiche', 
        'Date de naissance',
        'Si oui preciser la date du dernier don.',
        'Date de dernières règles (DDR) ',
        'Horodateur'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Créer une copie du DataFrame pour chaque type de donnée
    candidats_df = df.copy()
    
    # Effectuer le prétraitement spécifique aux candidats
    candidats_clean = df.copy()
    
    # Standardiser les noms de colonnes
    candidats_clean.columns = [col.strip() for col in candidats_clean.columns]
    
    # Ajouter l'éligibilité en code si la colonne existe
    eligibility_col = [col for col in candidats_clean.columns if 'LIGIBILIT' in col or 'ligibilit' in col.lower()]
    if eligibility_col:
        eligibility_map = {
            'Eligible': 1, 
            'Temporairement Non-eligible': 0, 
            'Définitivement non-eligible': -1
        }
        
        eligibility_col_name = eligibility_col[0]
        candidats_clean['eligibilite_code'] = candidats_clean[eligibility_col_name].map(eligibility_map)
    
    # Calculer l'âge si les dates de naissance existent
    if 'Date de naissance' in candidats_clean.columns and 'Date de remplissage de la fiche' in candidats_clean.columns:
        candidats_clean['age'] = (candidats_clean['Date de remplissage de la fiche'] - candidats_clean['Date de naissance']).dt.days / 365.25
    
    # Créer un DataFrame pour les donneurs (si pas de distinction claire dans le CSV)
    donneurs_clean = df.copy()
    
    # Créer un DataFrame pour les candidats avec âge
    candidats_age_clean = df.copy()
    if 'age' not in candidats_age_clean.columns and 'Date de naissance' in candidats_age_clean.columns:
        if 'Date de remplissage de la fiche' in candidats_age_clean.columns:
            candidats_age_clean['age'] = (candidats_age_clean['Date de remplissage de la fiche'] - candidats_age_clean['Date de naissance']).dt.days / 365.25
    
    # Créer un DataFrame combiné
    combined_df = candidats_clean.copy()
    
    return {
        'candidats': candidats_clean,
        'donneurs': donneurs_clean,
        'candidats_age': candidats_age_clean,
        'combined': combined_df
    }