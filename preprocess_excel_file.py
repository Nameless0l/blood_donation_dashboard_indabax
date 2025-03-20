import pandas as pd
import numpy as np
import os

def preprocess_excel_file():
    """
    Script de prétraitement pour le fichier Excel.
    Corrige les problèmes de type et sauvegarde un fichier CSV propre.
    """
    print("Démarrage du prétraitement...")
    input_file = "data/Updated Challenge dataset.xlsx"
    output_file = "data/preprocessed_data.csv"
    
    # Vérifier si le fichier d'entrée existe
    if not os.path.exists(input_file):
        print(f"ERREUR: Le fichier {input_file} n'existe pas!")
        return
    
    try:
        # Charger la feuille la plus complète
        sheet_name = 'Candidat au don 2019  (avec age'
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        print(f"Feuille '{sheet_name}' chargée avec succès. {len(df)} lignes trouvées.")
    except Exception as e:
        print(f"Erreur lors du chargement de la feuille spécifique: {str(e)}")
        # Essayer de charger la première feuille disponible
        df = pd.read_excel(input_file)
        print(f"Première feuille chargée à la place. {len(df)} lignes trouvées.")
    
    # Afficher les types initiaux pour le débogage
    print("\nTypes de données initiaux:")
    print(df.dtypes.head())
    
    # Créer une nouvelle version propre du DataFrame
    clean_df = pd.DataFrame()
    
    # Copier l'ID avec un nom standardisé
    if 'ID' in df.columns:
        clean_df['id_donneur'] = df['ID']
    else:
        # Générer des ID séquentiels si non présents
        clean_df['id_donneur'] = [f"DONOR_{i}" for i in range(1, len(df) + 1)]
    
    # Traiter l'âge - c'est la colonne la plus problématique
    if 'Age' in df.columns:
        # Convertir explicitement en numérique
        clean_df['age'] = pd.to_numeric(df['Age'], errors='coerce')
        # Vérifier les valeurs non numériques
        non_numeric = df.loc[pd.to_numeric(df['Age'], errors='coerce').isna(), 'Age']
        if len(non_numeric) > 0:
            print(f"Valeurs d'âge non numériques trouvées: {non_numeric.tolist()}")
        # Remplacer les valeurs manquantes par la médiane
        median_age = clean_df['age'].median()
        clean_df['age'] = clean_df['age'].fillna(median_age)
        # Convertir en entier
        clean_df['age'] = clean_df['age'].astype(int)
        print(f"Colonne 'age' traitée. Plage: {clean_df['age'].min()}-{clean_df['age'].max()}")
    
    # Traiter le genre
    if 'Genre_' in df.columns:
        clean_df['sexe'] = df['Genre_'].fillna('Non spécifié')
    elif 'Genre' in df.columns:
        clean_df['sexe'] = df['Genre'].fillna('Non spécifié')
    else:
        # Utiliser des valeurs aléatoires
        clean_df['sexe'] = np.random.choice(['Homme', 'Femme'], size=len(df))
    
    # Arrondissement et quartier
    arrond_cols = [col for col in df.columns if 'arrondissement' in col.lower()]
    if arrond_cols:
        clean_df['arrondissement'] = df[arrond_cols[0]].fillna('Non spécifié')
    else:
        clean_df['arrondissement'] = 'Non spécifié'
    
    quartier_cols = [col for col in df.columns if 'quartier' in col.lower()]
    if quartier_cols:
        clean_df['quartier'] = df[quartier_cols[0]].fillna('Non spécifié')
    else:
        clean_df['quartier'] = 'Non spécifié'
    
    # Profession
    prof_cols = [col for col in df.columns if 'profession' in col.lower()]
    if prof_cols:
        clean_df['profession'] = df[prof_cols[0]].fillna('Non spécifié')
    else:
        clean_df['profession'] = 'Non spécifié'
    
    # Conditions de santé - chercher toutes les colonnes contenant "raison" et "eligibilite"
    health_conditions = {}
    for col in df.columns:
        col_lower = col.lower().replace('é', 'e').replace('è', 'e')
        if 'raison' in col_lower and ('eligibilite' in col_lower or 'non-elig' in col_lower):
            # Extraire le nom de la condition
            if '[' in col and ']' in col:
                condition = col.split('[')[1].split(']')[0]
                # Normaliser le nom de la condition
                condition_key = condition.lower().replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
                # Extraire et convertir la valeur (0 ou 1)
                health_conditions[condition_key] = df[col].apply(
                    lambda x: 1 if isinstance(x, str) and x.lower() in ['oui', 'yes', 'true', '1'] else 0
                )
    
    # Ajouter les colonnes de santé
    for condition_key, values in health_conditions.items():
        clean_df[condition_key] = values
    
    # Créer une colonne condition_sante basée sur les drapeaux
    clean_df['condition_sante'] = 'Aucune'
    
    # Mapper les conditions avec des noms normalisés
    condition_mapping = {
        'porteurhivhbshcv': 'Porteur HIV/HBS/HCV',
        'diabetique': 'Diabétique', 
        'hypertendus': 'Hypertendu',
        'asthmatiques': 'Asthmatique',
        'cardiaque': 'Cardiaque',
        'drepanocytaire': 'Drépanocytaire'
    }
    
    # Appliquer le mapping
    for raw_key, display_name in condition_mapping.items():
        matching_keys = [k for k in health_conditions.keys() if raw_key in k.lower()]
        if matching_keys:
            key = matching_keys[0]
            clean_df.loc[clean_df[key] == 1, 'condition_sante'] = display_name
    
    # Éligibilité
    # Vérifier si des colonnes d'éligibilité explicites existent
    elig_cols = [col for col in df.columns if 'eligib' in col.lower()]
    if elig_cols:
        # Utiliser la première colonne d'éligibilité trouvée
        clean_df['eligible'] = df[elig_cols[0]].apply(
            lambda x: 1 if isinstance(x, str) and ('eligible' in str(x).lower() or 'oui' in str(x).lower()) else 0
        )
    else:
        # Déterminer l'éligibilité à partir des conditions de santé
        ineligible_conditions = ['porteurhivhbshcv', 'drepanocytaire']
        clean_df['eligible'] = 1  # Par défaut tous éligibles
        
        for condition in ineligible_conditions:
            matching_cols = [col for col in clean_df.columns if condition in col.lower()]
            if matching_cols:
                for col in matching_cols:
                    # Marquer comme non éligible si la condition est présente
                    clean_df.loc[clean_df[col] == 1, 'eligible'] = 0
    
    # S'assurer que eligible est un entier
    clean_df['eligible'] = clean_df['eligible'].astype(int)
    
    # Commentaires
    comment_cols = [col for col in df.columns if 'raison' in col.lower() and 'preciser' in col.lower()]
    if comment_cols:
        clean_df['commentaire'] = df[comment_cols[0]].fillna('')
    
    # Calculer les tranches d'âge (utile pour les visualisations)
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    clean_df['tranche_age'] = pd.cut(clean_df['age'], bins=bins, labels=labels, right=False)
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    # Sauvegarder en CSV pour éviter les problèmes d'Excel
    clean_df.to_csv(output_file, index=False)
    print(f"\nFichier prétraité sauvegardé: {output_file}")
    print(f"Colonnes dans le fichier prétraité: {clean_df.columns.tolist()}")
    print(f"Nombre d'enregistrements: {len(clean_df)}")
    print(f"Nombre de donneurs éligibles: {clean_df['eligible'].sum()}")

if __name__ == "__main__":
    preprocess_excel_file()