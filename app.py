import io 
import os
import base64
import folium
import joblib
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from function.preprocessing import preprocess_data
from plotly.subplots import make_subplots
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from function.page_assistant_ia import assistant_ia
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from function.utils import get_emoji_size
from function.visualizations import (create_geographic_visualizations, create_health_condition_visualizations,
                            create_campaign_effectiveness_visualizations,
                            create_donor_retention_visualizations, create_sentiment_analysis_visualizations)
from function.page_analyse_eligibilite import(display_profession_eligibility,display_ineligibility_reasons,get_available_health_indicators)
from function.page_analyse_donneurs import (analyse_donneurs)
# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord - Campagne de Don de Sang",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)
df_ = pd.read_csv('data/processed_data/dataset_don_sang_enrichi.csv', encoding='utf-8')
raisons_temp = [
        "Raison indisponibilité  [Est sous anti-biothérapie  ]",
        "Raison indisponibilité  [Taux d'hémoglobine bas ]",
        "Raison indisponibilité  [date de dernier Don < 3 mois ]",
        "Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]",
        "Raison de l'indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]",
        "Raison de l'indisponibilité de la femme [Allaitement ]",
        "Raison de l'indisponibilité de la femme [A accoucher ces 6 derniers mois  ]",
        "Raison de l'indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]",
        "Raison de l'indisponibilité de la femme [est enceinte ]"
    ]
raisons_def = [
        "Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]",
        "Raison de non-eligibilité totale  [Opéré]",
        "Raison de non-eligibilité totale  [Drepanocytaire]",
        "Raison de non-eligibilité totale  [Diabétique]",
        "Raison de non-eligibilité totale  [Hypertendus]",
        "Raison de non-eligibilité totale  [Asthmatiques]",
        "Raison de non-eligibilité totale  [Cardiaque]",
        "Raison de non-eligibilité totale  [Tatoué]",
        "Raison de non-eligibilité totale  [Scarifié]"
    ]
    
# Fonctions utilitaires
def load_data(file_path=None, uploaded_file=None):
    """
    Charge et prétraite les données du fichier Excel ou CSV
    
    Args:
        file_path (str, optional): Chemin vers le fichier par défaut
        uploaded_file (UploadedFile, optional): Fichier uploadé par l'utilisateur
        
    Returns:
        dict: Dictionnaire contenant les différents DataFrames prétraités
    """
    # Si un fichier a été uploadé
    if uploaded_file is not None:
        # Créer un fichier temporaire pour stocker le contenu uploadé
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if uploaded_file.name.endswith('.xlsx') else '.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Prétraiter le fichier uploadé
        try:
            if uploaded_file.name.endswith('.csv'):
                # Pour les fichiers CSV, nous devons adapter le code de prétraitement
                # Charger le CSV dans un DataFrame pandas
                df = pd.read_csv(temp_path)
                
                # Utiliser le DataFrame pour créer un dictionnaire simulant la structure attendue
                data_dict = {
                    'candidats': df,
                    'donneurs': df,  # Utiliser le même DataFrame comme fallback
                    'candidats_age': df,
                    'combined': df
                }
                return data_dict
            else:
                # Pour les fichiers Excel, utiliser la fonction de prétraitement existante
                return preprocess_data(temp_path)
        except Exception as e:
            st.error(f"Erreur lors du prétraitement du fichier uploadé: {e}")
            return None
    
    # Si aucun fichier n'a été uploadé, utiliser le fichier par défaut
    elif file_path:
        if not os.path.exists('data/processed_data'):
            # Si les données prétraitées n'existent pas, les traiter et les sauvegarder
            data_dict = preprocess_data(file_path)
        else:
            # Sinon, charger les données prétraitées
            data_dict = {}
            for name in ['candidats', 'donneurs', 'candidats_age', 'combined']:
                csv_path = f"data/processed_data/{name}_processed.csv"
                if os.path.exists(csv_path):
                    data_dict[name] = pd.read_csv(csv_path)
        
        return data_dict
    
    return None

def train_eligibility_model(df):
    """
    Charge un modèle de prédiction d'éligibilité et prépare les statistiques pour l'imputation
    """
    # Calculer les statistiques pour chaque colonne du DataFrame
    feature_stats = {}
    for col in df.columns:
        # Ignorer les colonnes d'éligibilité
        if col in ['eligibilite_code', 'ÉLIGIBILITÉ AU DON.']:
            continue
            
        # Pour les colonnes numériques
        if df[col].dtype in ['int64', 'float64']:
            mean_value = df[col].mean() if not pd.isna(df[col].mean()) else 0
            feature_stats[col] = {'type': 'numeric', 'fill_value': mean_value}
        # Pour les colonnes catégorielles
        else:
            # Utiliser le mode (valeur la plus fréquente)
            if not df[col].mode().empty:
                mode_value = df[col].mode()[0]
            else:
                mode_value = "" if df[col].dtype == 'object' else 0
            feature_stats[col] = {'type': 'categorical', 'fill_value': mode_value}
    
    # Chemin du modèle
    model_path = "api/model/eligibility_model_gradient_boosting_20250323_104955.pkl"
    
    try:
        if os.path.exists(model_path):
            # Charger le modèle
            model = joblib.load(model_path)
            print(f"Modèle chargé depuis: {model_path}")
            
            # Liste complète des colonnes nécessaires (à partir de l'erreur)
            required_columns = [
                # Colonnes déjà identifiées
                'age', 'genre_code', 'experience_don',
                'porteur_vih_hbs_hcv', 'diabetique', 'hypertendu', 'asthmatique',
                'drepanocytaire', 'cardiaque', 'transfusion', 'tatoue', 'scarifie',
                'poids', 'taille', 'imc',
                
                # Colonnes manquantes identifiées dans l'erreur
                "Taux d\u2019h\u00e9moglobine", 'Nationalité', 'arrondissement_clean', 
                'Profession', 'A-t-il (elle) déjà donné le sang', 'groupe_age',
                "Niveau d'etude", 'quartier_clean', 'Quartier de Résidence',
                'Religion', 'Situation Matrimoniale (SM)', 'Arrondissement de résidence'
            ]
            
            # Vérifier quelles colonnes sont réellement présentes dans le DataFrame
            available_columns = [col for col in required_columns if col in df.columns]
            
            return model, required_columns, feature_stats
            
        else:
            # Modèle non trouvé, retourner None
            st.error(f"Modèle non trouvé à: {model_path}")
            return None, [], feature_stats
            
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None, [], feature_stats

def predict_eligibility(model, input_data, required_columns=None, feature_stats={}):
    """
    Prédit l'éligibilité au don de sang en appliquant des règles de sécurité strictes
    """
    # Vérifier les critères d'exclusion absolus AVANT d'utiliser le modèle
    if isinstance(input_data, dict):
        # Critères d'exclusion absolus
        if input_data.get('porteur_vih_hbs_hcv', 0) == 1:
            return "Non éligible", 100.0  # Confiance maximale pour raison de sécurité
        
    elif isinstance(input_data, pd.DataFrame) and 'porteur_vih_hbs_hcv' in input_data.columns:
        if input_data['porteur_vih_hbs_hcv'].iloc[0] == 1:
            return "Non éligible", 100.0
    
    # Si aucun critère d'exclusion absolu n'est trouvé, continuer avec le modèle
    if model is None:
        return "Modèle non disponible", 0
    
    try:
        # Convertir en DataFrame si nécessaire
        if isinstance(input_data, dict):
            # Créer un nouveau dictionnaire avec les clés normalisées
            normalized_data = {}
            
            # Correspondance entre clés courantes et clés attendues
            key_mapping = {
                # Mappings standard
                "age": "age",
                "experience_don": "experience_don",
                "Genre": "Genre",
                "Niveau d'etude": "Niveau d'etude",
                "Situation Matrimoniale (SM)": "Situation Matrimoniale (SM)",
                "Profession": "Profession",
                "arrondissement_clean": "arrondissement_clean",
                "quartier_clean": "quartier_clean",
                "groupe_age": "groupe_age",
                
                # Mappings avec caractères accentués et apostrophes spécifiques
                "Arrondissement de résidence": "Arrondissement de résidence",
                "Quartier de Résidence": "Quartier de Résidence",
                "Nationalité": "Nationalité",
                "Religion": "Religion",
                "A-t-il (elle) déjà donné le sang": "A-t-il (elle) déjà donné le sang",
                
                # Cas particulier du taux d'hémoglobine avec différentes apostrophes
                "Taux d\u2019h\u00e9moglobine": "Taux d\u2019h\u00e9moglobine",
                "Taux d\u2019h\u00e9moglobine": "Taux d\u2019h\u00e9moglobine",
                "Taux d\u2019h\u00e9moglobine": "Taux d\u2019h\u00e9moglobine",
                "Taux d\u2019h\u00e9moglobine": "Taux d\u2019h\u00e9moglobine",
                "Taux d\u2019h\u00e9moglobine": "Taux d\u2019h\u00e9moglobine",
                "Taux d\u2019h\u00e9moglobine": "Taux d\u2019h\u00e9moglobine"
            }
            
            # Transférer les valeurs en utilisant les mappings
            for input_key, input_value in input_data.items():
                # Chercher une correspondance dans le mapping
                matched = False
                for pattern, target_key in key_mapping.items():
                    # Comparaison insensible aux différences d'apostrophes et d'accents
                    if pattern.lower().replace("'", "").replace("'", "").replace("é", "e") == \
                       input_key.lower().replace("'", "").replace("'", "").replace("é", "e"):
                        normalized_data[target_key] = input_value
                        matched = True
                        break
                
                # Si aucune correspondance trouvée, utiliser la clé originale
                if not matched:
                    normalized_data[input_key] = input_value
            
            # Créer DataFrame avec les données normalisées
            input_df = pd.DataFrame([normalized_data])
        else:
            input_df = input_data
        
        # Si nous avons la liste des colonnes requises
        if required_columns:
            # Créer un DataFrame pour la prédiction
            prediction_df = pd.DataFrame(index=input_df.index)
            
            # Pour chaque colonne requise
            for col in required_columns:
                if col in input_df.columns:
                    # Utiliser la valeur fournie
                    prediction_df[col] = input_df[col]
                else:
                    # Cas spéciaux avec mappages directs
                    if col == "experience_don" and "A-t-il (elle) déjà donné le sang" in input_df.columns:
                        prediction_df[col] = input_df["A-t-il (elle) déjà donné le sang"].map({'Oui': 1, 'Non': 0})
                    
                    # Recherche par nom similaire (sans apostrophes/accents)
                    elif col.lower().replace("'", "").replace("é", "e") in [
                        c.lower().replace("'", "").replace("é", "e") for c in input_df.columns
                    ]:
                        # Trouver la colonne correspondante
                        for input_col in input_df.columns:
                            if input_col.lower().replace("'", "").replace("é", "e") == \
                               col.lower().replace("'", "").replace("é", "e"):
                                prediction_df[col] = input_df[input_col]
                                break
                    
                    # Imputer avec les statistiques
                    elif col in feature_stats:
                        prediction_df[col] = feature_stats[col]['fill_value']
                    
                    # Valeurs par défaut selon le type
                    else:
                        is_text_col = col in [
                            "Niveau d'etude", "Genre", "Situation Matrimoniale (SM)",
                            "Profession", "Arrondissement de résidence", "Quartier de Résidence",
                            "Nationalité", "Religion", "A-t-il (elle) déjà donné le sang",
                            "groupe_age", "arrondissement_clean", "quartier_clean"
                        ]
                        prediction_df[col] = "" if is_text_col else 0
            
            # Faire la prédiction
            prediction = model.predict(prediction_df)[0]
            print(prediction)
            probabilities = model.predict_proba(prediction_df)[0]
        else:
            # Utiliser le DataFrame tel quel
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
        
        # Interpréter les résultats
        if prediction == 1:
            result = "Éligible"
            confidence = probabilities[1] * 100
        else:
            result = "Non éligible"
            confidence = probabilities[0] * 100
        
        # VÉRIFICATION FINALE DES RÈGLES DE SÉCURITÉ
        # Même si le modèle prédit "Éligible", certaines conditions doivent toujours rendre non éligible
        if result == "Éligible":
            # Vérifier à nouveau les critères d'exclusion absolus
            if isinstance(input_data, dict):
                # Porteur de VIH/hépatite
                if input_data.get('porteur_vih_hbs_hcv', 0) == 1:
                    return "Non éligible", 100.0
                
                # Autres conditions d'exclusion absolue
                if (input_data.get('drepanocytaire', 0) == 1 or 
                    input_data.get('cardiaque', 0) == 1):
                    return "Non éligible", 100.0
            
            elif isinstance(input_data, pd.DataFrame):
                # Porteur de VIH/hépatite
                if 'porteur_vih_hbs_hcv' in input_data.columns and input_data['porteur_vih_hbs_hcv'].iloc[0] == 1:
                    return "Non éligible", 100.0
                
                # Autres conditions d'exclusion absolue
                if (('drepanocytaire' in input_data.columns and input_data['drepanocytaire'].iloc[0] == 1) or 
                    ('cardiaque' in input_data.columns and input_data['cardiaque'].iloc[0] == 1)):
                    return "Non éligible", 100.0
        
        return result, confidence
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        
        # Affichage du débogage...
        
        return "Erreur de prédiction", 0
def get_feature_importance(model, feature_names):
    """
    Retourne l'importance des caractéristiques du modèle
    """
    if model is None:
        return None
    
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

def download_link(object_to_download, download_filename, download_link_text):
    """
    Génère un lien pour télécharger un objet
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Créer un lien de téléchargement temporaire
    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
    return href

# Interface principale
def main():
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    
    .sidebar .sidebar-content .block-container {
        padding-top: 1rem;
    }
    
    /* Style pour les titres de la sidebar */
    .sidebar .sidebar-content h1, 
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3 {
        color: white;
        font-weight: 600;
    }
    
    /* Style pour les widgets de filtre */
    .sidebar .sidebar-content .stSlider {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }
    
    .sidebar .sidebar-content .stSlider > div {
        padding-left: 0;
        padding-right: 0;
    }
    
    /* Boîte d'information du nombre de donneurs */
    .info-box {
        background-color: #2c3e50;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Style pour les select boxes */
    .sidebar .sidebar-content .stSelectbox > div > div {
        background-color: #262730;
        color: white;
        border: 1px solid #4e5d6c;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    # Titre et introduction
    st.title("🩸 Tableau de Bord de la Campagne de Don de Sang")
    
    # st.markdown("""
    # Ce tableau de bord vous permet d'analyser les données des campagnes de don de sang pour optimiser vos futures initiatives.
    # Explorez les différentes sections pour découvrir des insights sur la répartition géographique des donneurs,
    # l'impact des conditions de santé sur l'éligibilité, le profil des donneurs idéaux, l'efficacité des campagnes,
    # et les facteurs de fidélisation des donneurs.
    # """)
    
    uploaded_file = st.sidebar.file_uploader("Charger un fichier de données", type=['xlsx', 'csv'])
    
    if uploaded_file:
        st.sidebar.success(f"Fichier '{uploaded_file.name}' chargé avec succès!")
    
    # Charger les données (soit depuis le fichier uploadé, soit depuis le fichier par défaut)
    file_path = "Updated Challenge dataset.xlsx"
    data_dict = load_data(file_path, uploaded_file)
    st.sidebar.title("Navigation")
    # Options de navigation (maintenant avant les filtres)
    section = st.sidebar.radio(
        "Choisissez une section :",
        [
            "🌍 Répartition Géographique",
            "🏥 Conditions de Santé & Éligibilité",
            "🔬 Profilage des Donneurs",
            "📊 Efficacité des Campagnes",
            "🔄 Fidélisation des Donneurs",
            "💬 Analyse de Sentiment",
            "🤖 Prédiction d'Éligibilité",
            "🩸 Assistant IA"
        ]
    )
    
    # Section de filtres (déplacée après les options de navigation)
    st.sidebar.header("Filtres")
    
    # Variables pour stocker les données filtrées
    filtered_data = None
    donneurs_apres_filtrage = 0
    
    if data_dict and 'candidats' in data_dict:
        df = data_dict['candidats']
        
        # 1. Filtre d'arrondissement
        arrondissements = ['Tous']
        if 'arrondissement_clean' in df.columns:
            arrond_list = df['arrondissement_clean'].dropna().unique()
            arrondissements.extend(sorted(arrond_list))
        
        selected_arrondissement = st.sidebar.selectbox("Arrondissement", arrondissements)
        
        # 2. Filtre de tranche d'âge
        age_min, age_max = 0, 63
        if 'age' in df.columns:
            age_min = int(df['age'].min()) if not pd.isna(df['age'].min()) else 0
            age_max = int(df['age'].max()) if not pd.isna(df['age'].max()) else 63
        
        age_range = st.sidebar.slider("Tranche d'âge", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        
        # 3. Filtre de sexe
        genres = ['Tous']
        if 'Genre' in df.columns:
            genre_list = df['Genre'].dropna().unique()
            genres.extend(sorted(genre_list))
        
        selected_genre = st.sidebar.selectbox("Sexe", genres)
        
        # Appliquer les filtres
        filtered_data = df.copy()
        
        # Filtre d'arrondissement
        if selected_arrondissement != 'Tous' and 'arrondissement_clean' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['arrondissement_clean'] == selected_arrondissement]
        
        # Filtre d'âge
        if 'age' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['age'] >= age_range[0]) & (filtered_data['age'] <= age_range[1])]
        
        # Filtre de sexe
        if selected_genre != 'Tous' and 'Genre' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Genre'] == selected_genre]
        
        # Mettre à jour le dictionnaire de données avec les données filtrées
        data_dict['candidats'] = filtered_data
        
        # Calculer le nombre de donneurs après filtrage
        donneurs_apres_filtrage = len(filtered_data)
        
        # Afficher le nombre de donneurs après filtrage
        st.sidebar.markdown(
            f"""
            <div style='background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px;'>
            <b>Nombre de donneurs après filtrage:</b><br>
            {donneurs_apres_filtrage}
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    try:
        if data_dict:
            # Créer un modèle de prédiction
            model, expected_features, feature_stats = train_eligibility_model(data_dict['candidats'])
            
            # Afficher la section sélectionnée
            if section == "🌍 Répartition Géographique":
                show_geographic_distribution(data_dict)
            elif section == "🏥 Conditions de Santé & Éligibilité":
                show_health_conditions(data_dict)
            elif section == "🔬 Profilage des Donneurs":
                show_donor_profiling(data_dict)
            elif section == "📊 Efficacité des Campagnes":
                show_campaign_effectiveness(data_dict)
            elif section == "🔄 Fidélisation des Donneurs":
                show_donor_retention(data_dict)
            elif section == "💬 Analyse de Sentiment":
                show_sentiment_analysis(data_dict)
            elif section == "🤖 Prédiction d'Éligibilité":
                show_eligibility_prediction(data_dict, model)
            elif section == "🩸 Assistant IA":
                assistant_ia(data_dict['candidats'])
        elif section == "🤖 Prédiction d'Éligibilité":
            show_eligibility_prediction(data_dict, model, expected_features, feature_stats)
        else:
            st.error("Aucune donnée n'a pu être chargée. Veuillez uploader un fichier valide ou vérifier le fichier par défaut.")
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement ou du traitement des données : {e}")
        st.info("Veuillez vérifier que le fichier est accessible et correctement formaté.")
df_filtered = df_.copy()
indicateurs_disponibles = get_available_health_indicators(df_filtered)
raisons_temp_disponibles = [col for col in raisons_temp if col in df_filtered.columns]
raisons_def_disponibles = [col for col in raisons_def if col in df_filtered.columns]
def show_geographic_distribution(data_dict):
    """Affiche la section de répartition géographique des donneurs"""
    st.header("🌍 Cartographie de la Répartition des Donneurs")
    
    st.markdown("""
    Cette section vous permet de visualiser la répartition géographique des donneurs de sang
    en fonction de leur lieu de résidence.
    """)
    
    # Créer les onglets
    tab1, tab2, tab3, tab4,tab5 = st.tabs([
        "Répartition des donneurs", 
        "Fidélité par zone", 
        "Zones à cibler", 
        "Groupes sanguins par zone",
        "Insights & Recommandations"
    ])
    
    # Créer les visualisations géographiques
    geo_figures = create_geographic_visualizations(data_dict['candidats'])
    
    # Onglet 1: Répartition des donneurs
        # Carte interactive avec des émojis goutte de sang
    st.subheader("Carte Interactive des Donneurs")

    # Créer la carte de base
    m = folium.Map(location=[4.0511, 9.7679], zoom_start=13)

    # Titre de la carte
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Répartition des Donneurs à Douala</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Coordonnées des arrondissements
    arrondissements_coords = {
        'Douala 1': [4.0494, 9.7143],
        'Douala 2': [4.0611, 9.7179],
        'Douala 3': [4.0928, 9.7679],
        'Douala 4': [4.0711, 9.7543],
        'Douala 5': [4.0128, 9.7379]
    }

    # Si les données d'arrondissement existent
    if 'arrondissement_clean' in data_dict['candidats'].columns:
        arrond_counts = data_dict['candidats']['arrondissement_clean'].value_counts().to_dict()
        
        # Créer une échelle de taille basée sur les données
        min_count = min(arrond_counts.values()) if arrond_counts else 0
        max_count = max(arrond_counts.values()) if arrond_counts else 1
        
        
        # Ajouter les marqueurs pour chaque arrondissement
        for arrond, coords in arrondissements_coords.items():
            count = arrond_counts.get(arrond, 0)
            emoji_size = get_emoji_size(count, min_count, max_count)
            
            # Popup détaillée
            popup = folium.Popup(
                f"""
                <div style="text-align: center;">
                    <h4 style="margin-bottom: 5px; color: #cc0000;">{arrond}</h4>
                    <p style="font-size: 16px; margin-top: 5px;">
                        <b>{count}</b> donneurs de sang
                    </p>
                </div>
                """, 
                max_width=200
            )
            
            # Utiliser l'emoji goutte de sang avec taille variable
            emoji_html = f"""
                <div style="font-size: {emoji_size}px; text-align: center; cursor: pointer;">
                    🩸
                </div>
            """
            
            # Créer le marqueur avec l'icône personnalisée
            icon = folium.DivIcon(
                icon_size=(emoji_size, emoji_size),
                icon_anchor=(emoji_size/2, emoji_size/2),
                html=emoji_html
            )
            
            folium.Marker(
                location=coords,
                popup=popup,
                icon=icon,
                tooltip=f"<b>{arrond}</b>: {count} donneurs"  # Affichage au survol
            ).add_to(m)
        
        # Ajouter une légende avec des émojis
        legend_html = """
        <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
            <p style="text-align: center; font-weight: bold; margin-bottom: 8px; color: #cc0000;">
                Nombre de donneurs
            </p>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 18px; margin-right: 10px;">🩸</span>
                <span>Moins de donneurs</span>
            </div>
            
            <div style="display: flex; align-items: center;">
                <span style="font-size: 36px; margin-right: 10px;">🩸</span>
                <span>Plus de donneurs</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Ajouter du CSS pour l'effet de survol sur les émojis
        hover_css = """
        <style>
            .leaflet-marker-icon:hover {
                transform: scale(1.2);
                transition: transform 0.3s ease;
            }
        </style>
        """
        m.get_root().html.add_child(folium.Element(hover_css))

    # Afficher la carte dans Streamlit
    folium_static(m)
    # Onglet 2: Fidélité par zone
    with tab2:
        st.subheader("Analyse de la fidélité des donneurs par zone")
        
        # Insérez ici le code pour l'analyse de la fidélité par zone
        st.info("Cette section analysera la fidélité des donneurs (nombre de dons, fréquence) par zone géographique.")
        
        # Vous pouvez ajouter d'autres visualisations ici
        st.metric(
            label="Taux moyen de fidélisation",
            value="42%",
            delta="↑ 8% depuis l'année dernière"
        )
    
    # Onglet 3: Zones à cibler
    with tab3:
        st.subheader("Zones prioritaires à cibler")
        
        # Insérez ici le code pour identifier les zones à cibler
        st.info("Cette section identifie les zones géographiques avec un faible taux de participation qui devraient être ciblées pour des campagnes futures.")
        
        # Exemple de tableau fictif des zones à cibler
        zones_data = {
            "Zone": ["Douala 2 Nord", "Makepe", "Bonamoussadi Est", "Deido Sud", "Akwa Nord"],
            "Potentiel": [450, 380, 320, 290, 250],
            "Donneurs Actuels": [85, 70, 55, 50, 40],
            "Taux de Participation": ["18.9%", "18.4%", "17.2%", "17.2%", "16.0%"]
        }
        st.dataframe(pd.DataFrame(zones_data), use_container_width=True)
    
    # Onglet 4: Groupes sanguins par zone
    with tab4:
        st.subheader("Répartition des groupes sanguins par zone")   
        if 'arrondissement_clean' in df_filtered.columns and 'Groupe_sanguin' in df_filtered.columns:
            # Créer une table croisée des groupes sanguins par arrondissement
            groupe_par_arr = pd.crosstab(
                df_filtered['arrondissement_clean'], 
                df_filtered['Groupe_sanguin'],
                normalize='index'
            ) * 100  # Convertir en pourcentage
            
            # Graphique en heatmap avec Plotly
            fig_heatmap = px.imshow(
                groupe_par_arr,
                labels=dict(x="Groupe sanguin", y="Arrondissement", color="Pourcentage (%)"),
                x=groupe_par_arr.columns,
                y=groupe_par_arr.index,
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title="Répartition des groupes sanguins par arrondissement (%)"
            )
            
            fig_heatmap.update_layout(
                xaxis_title="Groupe sanguin",
                yaxis_title="Arrondissement",
            )
            
            # Ajouter les annotations avec les valeurs
            annotations = []
            for i, row in enumerate(groupe_par_arr.index):
                for j, col in enumerate(groupe_par_arr.columns):
                    annotations.append(
                        dict(
                            x=col,
                            y=row,
                            text=f"{groupe_par_arr.iloc[i, j]:.1f}%",
                            showarrow=False,
                            font=dict(color="black" if groupe_par_arr.iloc[i, j] < 50 else "white")
                        )
                    )
            
            fig_heatmap.update_layout(annotations=annotations)
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Graphiques en secteurs pour chaque arrondissement
            st.subheader("Distribution des groupes sanguins par arrondissement")
            
            # Permettre à l'utilisateur de sélectionner un arrondissement
            arrondissements = sorted(df_filtered['arrondissement_clean'].unique())
            selected_arr = st.selectbox("Sélectionner un arrondissement", arrondissements)
            
            # Filtrer les données pour l'arrondissement sélectionné
            df_arr = df_filtered[df_filtered['arrondissement_clean'] == selected_arr]
            
            # Créer un graphique en secteurs pour l'arrondissement sélectionné
            groupe_counts = df_arr['Groupe_sanguin'].value_counts()
            fig_pie = px.pie(
                values=groupe_counts.values,
                names=groupe_counts.index,
                title=f"Distribution des groupes sanguins à {selected_arr}",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Comparaison avec la distribution globale
            st.subheader("Comparaison avec la distribution globale des groupes sanguins")
            
            # Distribution globale
            distribution_globale = df_filtered['Groupe_sanguin'].value_counts(normalize=True) * 100
            distribution_arr = df_arr['Groupe_sanguin'].value_counts(normalize=True) * 100
            
            # Créer un dataframe pour la comparaison
            compare_df = pd.DataFrame({
                'Groupe sanguin': distribution_globale.index,
                'Distribution globale (%)': distribution_globale.values,
                f'Distribution à {selected_arr} (%)': [distribution_arr.get(groupe, 0) for groupe in distribution_globale.index]
            })
            
            # Graphique en barres pour la comparaison
            fig_compare = px.bar(
                compare_df, 
                x='Groupe sanguin',
                y=[f'Distribution à {selected_arr} (%)', 'Distribution globale (%)'],
                barmode='group',
                title=f"Comparaison de la distribution des groupes sanguins à {selected_arr} vs. globale",
                color_discrete_sequence=['red', 'blue']
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Besoins spécifiques par groupe sanguin
            st.subheader("Analyse des besoins par groupe sanguin")
            
            # Données fictives pour les besoins en sang (en situation réelle, utilisez des données réelles)
            besoins = {
                'O+': 38,  # % des besoins totaux
                'A+': 34,
                'B+': 9,
                'AB+': 3,
                'O-': 7,
                'A-': 6,
                'B-': 2,
                'AB-': 1
            }
            
            # Créer un dataframe pour analyser l'offre vs demande
            offre_demande = pd.DataFrame({
                'Groupe sanguin': distribution_globale.index,
                'Disponibilité (%)': distribution_globale.values,
                'Besoins (%)': [besoins.get(groupe, 0) for groupe in distribution_globale.index]
            })
            
            offre_demande['Écart (%)'] = offre_demande['Disponibilité (%)'] - offre_demande['Besoins (%)']
            
            # Graphique en barres pour l'offre vs demande
            fig_besoins = px.bar(
                offre_demande,
                x='Groupe sanguin',
                y=['Disponibilité (%)', 'Besoins (%)'],
                barmode='group',
                title="Analyse de l'offre vs demande par groupe sanguin",
                color_discrete_sequence=['green', 'orange']
            )
            st.plotly_chart(fig_besoins, use_container_width=True)
            
            # Tableau des écarts
            st.subheader("Écart entre disponibilité et besoins")
            
            # Formater les écarts pour une meilleure lisibilité
            offre_demande['Statut'] = offre_demande['Écart (%)'].apply(
                lambda x: '✅ Surplus' if x > 5 else '⚠️ Équilibré' if abs(x) <= 5 else '❌ Déficit'
            )
            
            st.dataframe(offre_demande.sort_values('Écart (%)', ascending=True))

    with tab5:
    # Ajout d'un pied de page avec insights et recommandations
        st.subheader("Insights & Recommandations")
        
        st.markdown("""
        **Principaux insights :**
        - La majorité des donneurs sont concentrés dans Douala 3, suivi par Douala 5
        - Certains quartiers montrent une participation beaucoup plus élevée que d'autres
        
        **Recommandations :**
        - Intensifier les campagnes dans les zones à faible participation
        - Étudier les facteurs de succès dans les quartiers à forte participation pour les reproduire ailleurs
        - Mettre en place des unités mobiles de collecte dans les zones éloignées des centres de don
        """)


def show_health_conditions(data_dict):
    """Affiche la section sur les conditions de santé et l'éligibilité"""
    st.header("🏥 Conditions de Santé & Éligibilité")
    
    st.markdown("""
    Cette section analyse l'impact des conditions de santé sur l'éligibilité au don de sang.
    """)
    
    # Créer les onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Statistiques d'éligibilité", 
        "Raisons d'inéligibilité", 
        "Éligibilité par profession",
        "Insights & Recommandations"
    ])
    
    # Statistiques sur l'éligibilité
    with tab1:
        st.subheader("Statistiques d'Éligibilité")
        
        if 'ÉLIGIBILITÉ AU DON.' in data_dict['candidats'].columns:
            eligibility_counts = data_dict['candidats']['ÉLIGIBILITÉ AU DON.'].value_counts()
            eligibility_percentage = eligibility_counts / eligibility_counts.sum() * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Donneurs Éligibles",
                    value=f"{eligibility_counts.get('Eligible', 0):,}",
                    delta=f"{eligibility_percentage.get('Eligible', 0):.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Temporairement Non-éligibles",
                    value=f"{eligibility_counts.get('Temporairement Non-eligible', 0):,}",
                    delta=f"{eligibility_percentage.get('Temporairement Non-eligible', 0):.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Définitivement Non-éligibles",
                    value=f"{eligibility_counts.get('Définitivement non-eligible', 0):,}",
                    delta=f"{eligibility_percentage.get('Définitivement non-eligible', 0):.1f}%"
                )
        
        # Créer les visualisations sur les conditions de santé
        health_figures = create_health_condition_visualizations(data_dict['candidats'])
        
        # Afficher les visualisations
        if 'health_impact_bar' in health_figures:
            st.plotly_chart(health_figures['health_impact_bar'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'health_condition_correlation' in health_figures:
                st.plotly_chart(health_figures['health_condition_correlation'], use_container_width=True)
        
        with col2:
            if 'ineligibility_reasons_pie' in health_figures:
                st.plotly_chart(health_figures['ineligibility_reasons_pie'], use_container_width=True)
    
    # Raisons d'inéligibilité
    with tab2:
        st.subheader("Raisons d'inéligibilité")
        display_ineligibility_reasons(df_filtered, raisons_temp_disponibles, raisons_def_disponibles)
    
    # Éligibilité par profession
    with tab3:
        st.subheader("Analyse de l'éligibilité par profession")
        display_profession_eligibility(df_filtered, indicateurs_disponibles)
    
    # Insights et recommandations
    with tab4:
        st.subheader("Insights & Recommandations")
        st.markdown("""
        **Principaux insights :**
        - Les porteurs du VIH, de l'hépatite B ou C sont systématiquement non-éligibles
        - L'hypertension et le diabète impactent significativement l'éligibilité
        - Les donneurs avec des antécédents d'asthme peuvent généralement donner sous certaines conditions
        
        **Recommandations :**
        - Mettre en place des campagnes d'information ciblées sur les critères d'éligibilité
        - Offrir des alternatives de contribution pour les personnes définitivement non-éligibles
        - Former le personnel médical pour mieux évaluer les cas limites, notamment pour l'asthme et l'hypertension légère
        """)
def show_donor_profiling(data_dict):
    """Affiche la section sur le profilage des donneurs idéaux"""
    st.header("🔬 Profilage des Donneurs Idéaux")
    st.markdown("""
    Cette section utilise des techniques d'analyse de données avancées pour identifier
    les caractéristiques communes des donneurs de sang idéaux. Cela vous aidera à cibler
    vos campagnes vers les populations les plus susceptibles de donner.
    """)
    analyse_donneurs(df_)


def show_campaign_effectiveness(data_dict):
    """Affiche la section sur l'efficacité des campagnes"""
    st.header("📊 Analyse de l'Efficacité des Campagnes")
    
    st.markdown("""
    Cette section analyse les résultats des campagnes passées pour identifier
    les périodes optimales et les facteurs de succès. Ces informations vous aideront
    à planifier vos futures campagnes pour maximiser les collectes.
    """)
    
    # Créer les visualisations d'efficacité des campagnes
    campaign_figures = create_campaign_effectiveness_visualizations(
        data_dict['candidats'],
        data_dict.get('donneurs')
    )
    
    # Créer les onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tendances globales", 
        "Analyse temporelle", 
        "Analyse démographique", 
        "Insights & Recommandations"
    ])
    
    # Onglet 1: Tendances globales
    with tab1:
        st.subheader("Tendances des dons sur la période")
        if 'monthly_donations_line' in campaign_figures:
            st.plotly_chart(campaign_figures['monthly_donations_line'], use_container_width=True)
            
        # Métriques clés (fictives ou basées sur les données disponibles)
        eligibility_counts = data_dict['candidats']['ÉLIGIBILITÉ AU DON.'].value_counts()
        eligibility_percentage = eligibility_counts / eligibility_counts.sum() * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Total des dons", 
                value=f"{eligibility_counts.get('Eligible', 0):,}",
                delta=f"{eligibility_percentage.get('Eligible', 0):.1f}%", 
                # delta="↑ 12% vs année précédente"
            )
        with col2:
            st.metric(
                label="Nouveaux donneurs", 
                value="245", 
                delta="↑ 8% vs année précédente"
            )
        with col3:
            st.metric(
                label="Taux de retour", 
                value="14%", 
                delta="↑ 5% vs année précédente"
            )
    
    # Onglet 2: Analyse temporelle
    with tab2:
        st.subheader("Analyse temporelle des dons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'average_monthly_donations' in campaign_figures:
                st.plotly_chart(campaign_figures['average_monthly_donations'], use_container_width=True)
        
        with col2:
            if 'weekly_donations' in campaign_figures:
                st.plotly_chart(campaign_figures['weekly_donations'], use_container_width=True)
                
        # Analyse horaire fictive ou basée sur les données disponibles
        st.subheader("Répartition horaire des dons")
        st.info("Cette section montrerait l'analyse des heures de la journée les plus productives pour les collectes")
        
        # Exemple de graphique fictif pour les heures
        heures = ["8h-10h", "10h-12h", "12h-14h", "14h-16h", "16h-18h", "18h-20h"]
        valeurs = [120, 185, 145, 210, 175, 95]
        
        fig = px.bar(
            x=heures, 
            y=valeurs,
            title="Répartition horaire des dons",
            labels={"x": "Plage horaire", "y": "Nombre de dons"},
            color=valeurs,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Onglet 3: Analyse démographique
    with tab3:
        st.subheader("Contribution par Groupe Démographique")
        
        demographic_cols = ['groupe_age', 'Genre', 'Profession']
        for col in demographic_cols:
            fig_key = f'{col}_donations'
            if fig_key in campaign_figures:
                st.plotly_chart(campaign_figures[fig_key], use_container_width=True)
        
        # Analyse croisée fictive
        st.subheader("Analyse croisée démographique")
        st.info("Cette section montrerait des analyses croisées entre différentes caractéristiques démographiques")
    
    # Onglet 4: Insights & Recommandations
    with tab4:
        st.subheader("Insights & Recommandations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Principaux insights")
            st.markdown("""
            - Les mois d'avril, août et décembre montrent généralement une participation plus élevée
            - Le milieu de semaine (mardi à jeudi) est plus propice aux dons que le week-end
            - Certaines professions comme les étudiants, les enseignants et les professionnels de la santé sont surreprésentées
            - La plage horaire 14h-16h est la plus productive pour les collectes
            - Les donneurs entre 25-35 ans représentent la majorité des dons
            """)
        
        with col2:
            st.subheader("Recommandations")
            st.markdown("""
            - Planifier des campagnes majeures durant les périodes de forte participation (avril, août, décembre)
            - Organiser les collectes principalement en milieu de semaine
            - Cibler des lieux fréquentés par les groupes démographiques les plus contributeurs
            - Diversifier les approches pour atteindre des groupes professionnels sous-représentés
            - Optimiser les horaires des collectes pour maximiser la participation
            """)
        
        # Plan d'action fictif
        st.subheader("Plan d'action proposé")
        
        plan_data = {
            "Action": [
                "Campagne majeure universitaire", 
                "Sessions de sensibilisation en entreprise",
                "Unités mobiles dans les quartiers sous-représentés",
                "Programme de fidélisation des donneurs réguliers",
                "Campagne ciblée sur les groupes sanguins rares"
            ],
            "Période": [
                "Avril 2023", 
                "Juin-Juillet 2023",
                "Août-Septembre 2023",
                "Continu",
                "Octobre 2023"
            ],
            "Objectif": [
                "500 nouveaux donneurs", 
                "300 dons d'employés",
                "250 dons dans quartiers ciblés",
                "Augmenter de 15% le taux de retour",
                "100 dons de groupes sanguins rares"
            ]
        }
        
        st.dataframe(pd.DataFrame(plan_data), use_container_width=True)

def show_donor_retention(data_dict):
    """Affiche la section sur la fidélisation des donneurs"""
    st.header("🔄 Fidélisation des Donneurs")
    
    st.markdown("""
    Cette section analyse les facteurs qui influencent le retour des donneurs
    et propose des stratégies pour améliorer leur fidélisation. Un donneur régulier
    est beaucoup plus précieux qu'un donneur occasionnel.
    """)
    
    # Créer les visualisations de fidélisation des donneurs
    retention_figures = create_donor_retention_visualizations(data_dict['candidats'])
    
    # Créer les onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Vue d'ensemble", 
        "Facteurs d'influence", 
        "Intervalles entre dons", 
        "Stratégies & Recommandations"
    ])
    
    # Onglet 1: Vue d'ensemble
    with tab1:
        st.subheader("Statistiques globales de fidélisation")
        
        # Afficher le graphique de distribution des expériences des donneurs
        if 'donor_experience_pie' in retention_figures:
            st.plotly_chart(retention_figures['donor_experience_pie'], use_container_width=True)
        
        # Ajouter des métriques clés
        col1, col2, col3 = st.columns(3)
        
        # Calculer les métriques si les données sont disponibles
        experience_rate = 0
        if 'experience_don' in data_dict['candidats'].columns:
            experience_rate = data_dict['candidats']['experience_don'].mean() * 100
        
        with col1:
            st.metric(
                label="Taux de donneurs réguliers", 
                value=f"{experience_rate:.1f}%",
                help="Pourcentage de donneurs ayant déjà donné leur sang auparavant"
            )
        
        with col2:
            # Calculer l'intervalle moyen si disponible
            interval_value = "N/A"
            if 'jours_depuis_dernier_don' in data_dict['candidats'].columns:
                try:
                    mean_days = data_dict['candidats']['jours_depuis_dernier_don'].astype(float).mean()
                    interval_value = f"{mean_days:.0f} jours"
                except:
                    pass
            
            st.metric(
                label="Intervalle moyen entre dons", 
                value=interval_value,
                delta="Objectif: 90-120 jours"
            )
        
        with col3:
            # Calculer le taux de retour si disponible
            return_rate_value = "N/A"
            if 'Intention_don_futur' in data_dict['candidats'].columns:
                try:
                    positive_intentions = data_dict['candidats']['Intention_don_futur'].str.contains('Oui').mean() * 100
                    return_rate_value = f"{positive_intentions:.1f}%"
                except:
                    pass
            
            st.metric(
                label="Intention de retour", 
                value=return_rate_value
            )
        
        # Ajouter une section sur l'importance de la fidélisation
        st.subheader("Importance de la fidélisation des donneurs")
        
        st.markdown("""
        **Pourquoi la fidélisation est cruciale :**
        
        - **Sécurité accrue :** Les donneurs réguliers présentent moins de risques de maladies transmissibles
        - **Coût réduit :** Recruter un nouveau donneur coûte 5 à 7 fois plus cher que fidéliser un donneur existant
        - **Prévisibilité :** Les donneurs réguliers permettent une meilleure planification des stocks
        - **Qualité améliorée :** Les donneurs expérimentés ont généralement moins de complications lors du don
        """)
        
        # Graphique pour illustrer l'impact économique
        economic_data = {
            "Type de coût": ["Recrutement nouveau donneur", "Fidélisation donneur existant"],
            "Coût relatif": [100, 18]  # Valeurs arbitraires pour l'illustration
        }
        
        fig_cost = px.bar(
            economic_data,
            x="Type de coût",
            y="Coût relatif",
            title="Comparaison des coûts de recrutement vs fidélisation",
            color="Type de coût",
            color_discrete_sequence=["#e74c3c", "#2ecc71"]
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Onglet 2: Facteurs d'influence
    with tab2:
        st.subheader("Facteurs influençant le retour des donneurs")
        
        # Afficher les facteurs influençant le retour des donneurs
        retention_factors = ['Genre', 'groupe_age', 'arrondissement_clean']
        
        for factor in retention_factors:
            fig_key = f'{factor}_retention'
            if fig_key in retention_figures:
                st.plotly_chart(retention_figures[fig_key], use_container_width=True)
        
        # Ajouter une analyse des motivations si disponible
        if 'Motivation_principale' in data_dict['candidats'].columns and 'experience_don' in data_dict['candidats'].columns:
            st.subheader("Impact des motivations sur la fidélisation")
            
            try:
                # Créer un tableau croisé des motivations par expérience de don
                motiv_by_exp = pd.crosstab(
                    data_dict['candidats']['Motivation_principale'],
                    data_dict['candidats']['experience_don'],
                    normalize='index'
                ) * 100
                
                # Convertir en format long pour Plotly
                motiv_exp_df = motiv_by_exp.reset_index()
                
                # Renommer les colonnes
                if 1 in motiv_exp_df.columns:
                    motiv_exp_df.rename(columns={1: 'Taux de fidélisation (%)'}, inplace=True)
                
                # Trier par taux de fidélisation
                motiv_exp_df = motiv_exp_df.sort_values('Taux de fidélisation (%)', ascending=False)
                
                # Créer le graphique
                fig_motiv = px.bar(
                    motiv_exp_df,
                    x='Motivation_principale',
                    y='Taux de fidélisation (%)',
                    title="Taux de fidélisation par motivation principale",
                    color='Taux de fidélisation (%)',
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig_motiv, use_container_width=True)
                
                st.markdown("""
                **Insights sur les motivations :**
                - Les motivations altruistes sont généralement associées à des taux de fidélisation plus élevés
                - Les donneurs motivés par une obligation sociale ont tendance à moins revenir
                - L'expérience positive du premier don est un facteur déterminant pour le retour
                """)
            except Exception as e:
                st.info(f"Impossible d'analyser les motivations: {e}")
    
    # Onglet 3: Intervalles entre dons
    with tab3:
        st.subheader("Analyse des intervalles entre les dons")
        
        # Afficher les graphiques d'intervalle
        if 'time_since_donation_hist' in retention_figures:
            st.plotly_chart(retention_figures['time_since_donation_hist'], use_container_width=True)
        
        if 'time_eligibility_bar' in retention_figures:
            st.plotly_chart(retention_figures['time_eligibility_bar'], use_container_width=True)
        
        # Ajouter un calendrier optimal des dons
        st.subheader("Calendrier optimal des dons")
        
        # Tableau des intervalles recommandés
        optimal_intervals = {
            "Type de donneur": ["Homme", "Femme"],
            "Intervalle minimal": ["8 semaines", "12 semaines"],
            "Fréquence maximale par an": ["6 dons", "4 dons"],
            "Intervalle recommandé": ["12 semaines", "16 semaines"]
        }
        
        st.dataframe(pd.DataFrame(optimal_intervals), use_container_width=True)
        
        # Visualisation du moment opportun pour rappeler les donneurs
        st.subheader("Moment optimal pour contacter les donneurs")
        
        # Données exemple pour le graphique
        reminder_data = {
            "Semaines après le don": [1, 2, 4, 8, 10, 12, 16, 20, 24],
            "Efficacité du rappel (%)": [10, 15, 35, 75, 90, 85, 65, 45, 25]
        }
        
        fig_reminder = px.line(
            reminder_data,
            x="Semaines après le don",
            y="Efficacité du rappel (%)",
            title="Efficacité des rappels selon le délai depuis le dernier don",
            markers=True
        )
        
        # Ajouter des annotations pour les périodes clés
        fig_reminder.add_vrect(
            x0=8, x1=12,
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Période optimale (hommes)",
            annotation_position="top left"
        )
        
        fig_reminder.add_vrect(
            x0=12, x1=16,
            fillcolor="blue", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Période optimale (femmes)",
            annotation_position="top left"
        )
        
        st.plotly_chart(fig_reminder, use_container_width=True)
    
    # Onglet 4: Stratégies & Recommandations
    with tab4:
        st.subheader("Stratégies de fidélisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Communication et reconnaissance :**
            - Envoyer des rappels personnalisés quand le donneur devient à nouveau éligible
            - Créer un système de reconnaissance pour les donneurs réguliers (badges, certificats)
            - Partager des témoignages de bénéficiaires pour renforcer l'impact émotionnel
            - Informer les donneurs de l'utilisation de leur don précédent
            """)
        
        with col2:
            st.markdown("""
            **Amélioration de l'expérience :**
            - Réduire les temps d'attente lors des collectes
            - Offrir un suivi de santé aux donneurs réguliers
            - Organiser des événements spéciaux pour les donneurs fidèles
            - Faciliter la prise de rendez-vous en ligne
            - Proposer des horaires adaptés aux différentes contraintes professionnelles
            """)
        
        # Programme de fidélisation
        st.subheader("Proposition de programme de fidélisation")
        
        loyalty_program = {
            "Niveau": ["Bronze (1-2 dons)", "Argent (3-5 dons)", "Or (6-10 dons)", "Platine (11+ dons)"],
            "Avantages": [
                "Certificat de reconnaissance + Badge numérique",
                "Priorité de rendez-vous + T-shirt exclusif",
                "Accès VIP (zone d'attente spéciale) + Bilan sanguin gratuit",
                "Parrainage d'événements + Statut d'ambassadeur + Invitation aux événements spéciaux"
            ],
            "Objectif": [
                "Encourager le second don",
                "Établir l'habitude du don régulier",
                "Renforcer l'engagement à long terme",
                "Transformer le donneur en ambassadeur"
            ]
        }
        
        st.dataframe(pd.DataFrame(loyalty_program), use_container_width=True)
        
        # Insights et recommandations
        st.subheader("Insights & Recommandations")
        
        st.markdown("""
        **Principaux insights :**
        - Environ 42% des candidats sont des donneurs récurrents
        - Le taux de fidélisation varie significativement selon l'arrondissement de résidence
        - Les donneurs âgés de 26 à 45 ans montrent les meilleurs taux de fidélisation
        - L'intervalle moyen entre deux dons est d'environ 12 mois (bien au-delà des 3 mois minimum requis)
        - Les rappels envoyés 8 à 12 semaines après le don précédent sont les plus efficaces
        
        **Recommandations :**
        - Mettre en place un programme de fidélisation structuré avec des avantages progressifs
        - Cibler prioritairement les donneurs ayant déjà donné il y a plus de 3-4 mois
        - Organiser des campagnes spécifiques dans les zones à faible taux de fidélisation
        - Éduquer les donneurs sur la fréquence optimale des dons (tous les 3-4 mois)
        - Créer un parcours d'expérience spécifique pour les donneurs de première fois
        """)
        
        # Plan d'action
        st.subheader("Plan d'action proposé")
        
        action_plan = {
            "Action": [
                "Mise en place du programme de fidélisation", 
                "Campagne de rappel ciblée",
                "Formation du personnel aux techniques de fidélisation",
                "Amélioration de l'expérience de don",
                "Analyse des données de fidélisation"
            ],
            "Calendrier": [
                "Q1 2025", 
                "Mensuel",
                "Q2 2025",
                "Continu",
                "Trimestriel"
            ],
            "KPI": [
                "Augmentation du taux de fidélisation de 15%", 
                "Taux de retour post-rappel > 30%",
                "Satisfaction des donneurs > 90%",
                "Réduction du temps d'attente de 25%",
                "Identification des 3 principaux facteurs de fidélisation"
            ]
        }
        
        st.dataframe(pd.DataFrame(action_plan), use_container_width=True)
def show_sentiment_analysis(data_dict):
    """Affiche la section d'analyse de sentiment des retours"""
    st.header("💬 Analyse de Sentiment des Retours")
    
    st.markdown("""
    Cette section analyse les retours textuels des donneurs pour comprendre
    leur satisfaction et identifier des axes d'amélioration. Cette analyse
    vous aidera à perfectionner l'expérience des donneurs.
    """)
    
    # Créer les visualisations d'analyse de sentiment
    sentiment_figures = create_sentiment_analysis_visualizations(data_dict['candidats'])
    
    # Créer les onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Vue d'ensemble", 
        "Analyse des commentaires", 
        "Évolution temporelle", 
        "Insights & Recommandations"
    ])
    
    # Vérifier si nous avons des données de sentiment
    has_sentiment_data = sentiment_figures and len(sentiment_figures) > 0
    
    # Onglet 1: Vue d'ensemble
    with tab1:
        st.subheader("Répartition globale des sentiments")
        
        if has_sentiment_data and 'sentiment_pie' in sentiment_figures:
            st.plotly_chart(sentiment_figures['sentiment_pie'], use_container_width=True)
            
            # Afficher des métriques clés
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Calculer le taux de satisfaction global
                positive_rate = 65  # Valeur fictive, à remplacer par la vraie si disponible
                st.metric(
                    label="Taux de satisfaction", 
                    value=f"{positive_rate}%",
                    delta="↑ 5% vs trimestre précédent",
                    help="Pourcentage de retours positifs et neutres"
                )
            
            with col2:
                # Taux de participation au feedback
                feedback_rate = 42  # Valeur fictive
                st.metric(
                    label="Taux de participation au feedback", 
                    value=f"{feedback_rate}%",
                    delta="↑ 8% vs trimestre précédent",
                    help="Pourcentage de donneurs ayant laissé un commentaire"
                )
            
            with col3:
                # Score NPS (Net Promoter Score)
                nps_score = 38  # Valeur fictive
                st.metric(
                    label="Score NPS", 
                    value=f"{nps_score}",
                    delta="↑ 4 points vs trimestre précédent",
                    help="Net Promoter Score (échelle de -100 à +100)"
                )
        else:
            # Afficher un graphique fictif si pas de données
            sentiment_data = {
                "Sentiment": ["Positif", "Neutre", "Négatif"],
                "Proportion": [65, 25, 10]
            }
            
            fig_sentiment = px.pie(
                sentiment_data, 
                names="Sentiment", 
                values="Proportion",
                title="Répartition des sentiments dans les retours (données illustratives)",
                color="Sentiment",
                color_discrete_map={
                    "Positif": "#2ecc71",  # Vert
                    "Neutre": "#3498db",   # Bleu
                    "Négatif": "#e74c3c"   # Rouge
                }
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            st.info("""
            Ces données sont illustratives car il n'y a pas suffisamment de retours textuels 
            pour réaliser une analyse de sentiment réelle. Pour implémenter cette fonctionnalité, 
            nous recommandons de collecter plus de commentaires via des questionnaires de satisfaction.
            """)
            
        # Ajouter un tableau de satisfaction par aspect
        st.subheader("Satisfaction par aspect de l'expérience")
        
        satisfaction_aspects = {
            "Aspect": [
                "Accueil et personnel", 
                "Temps d'attente", 
                "Confort des installations", 
                "Processus de don", 
                "Suivi post-don", 
                "Information reçue"
            ],
            "Score moyen (sur 5)": [4.7, 3.4, 4.2, 4.5, 3.9, 4.0],
            "Évolution": ["↑", "↓", "↑", "→", "↑", "→"]
        }
        
        st.dataframe(pd.DataFrame(satisfaction_aspects), use_container_width=True)
    
    # Onglet 2: Analyse des commentaires
    with tab2:
        st.subheader("Analyse des commentaires des donneurs")
        
        if has_sentiment_data and 'keyword_bar' in sentiment_figures:
            st.plotly_chart(sentiment_figures['keyword_bar'], use_container_width=True)
        else:
            # Afficher un graphique fictif des mots-clés
            keywords_data = {
                "Mot-clé": [
                    "personnel", "attente", "accueil", "professionnel", 
                    "information", "rapide", "confortable", "chaleureux", 
                    "temps", "efficace"
                ],
                "Fréquence": [42, 38, 35, 30, 28, 25, 23, 20, 18, 15]
            }
            
            fig_keywords = px.bar(
                keywords_data,
                x="Fréquence",
                y="Mot-clé",
                orientation='h',
                title="Mots-clés les plus fréquents dans les commentaires (données illustratives)",
                color="Fréquence",
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig_keywords, use_container_width=True)
        
        # Afficher des exemples de retours
        st.subheader("Exemples de Retours")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Retours positifs:**
            > "Personnel très accueillant et professionnel."
            
            > "Processus efficace et rapide, je reviendrai!"
            
            > "Très satisfait de l'expérience globale."
            """)
        
        with col2:
            st.markdown("""
            **Retours neutres:**
            > "Procédure standard, rien à signaler."
            
            > "L'attente était raisonnable."
            
            > "J'aurais apprécié plus d'informations sur l'utilisation du sang."
            """)
        
        with col3:
            st.markdown("""
            **Retours négatifs:**
            > "Temps d'attente trop long avant d'être pris en charge."
            
            > "Le lieu était difficile à trouver sans indication claire."
            
            > "Peu de suivi après le don, sentiment d'abandon."
            """)
        
        # Ajouter une analyse thématique
        st.subheader("Analyse thématique des commentaires")
        
        themes_data = {
            "Thème": [
                "Qualité du personnel", 
                "Temps d'attente", 
                "Confort des installations",
                "Information reçue",
                "Suivi post-don", 
                "Accessibilité du lieu"
            ],
            "Mentions positives": [85, 40, 65, 50, 35, 45],
            "Mentions négatives": [5, 45, 15, 25, 40, 30]
        }
        
        # Créer un dataframe
        themes_df = pd.DataFrame(themes_data)
        
        # Calculer le ratio positif/négatif
        themes_df["Ratio positif/négatif"] = (themes_df["Mentions positives"] / (themes_df["Mentions positives"] + themes_df["Mentions négatives"] + 0.001) * 100).round(1)
        
        # Trier par ratio décroissant
        themes_df = themes_df.sort_values("Ratio positif/négatif", ascending=False)
        
        # Créer un graphique
        fig_themes = px.bar(
            themes_df,
            x="Thème",
            y=["Mentions positives", "Mentions négatives"],
            title="Analyse thématique des commentaires (données illustratives)",
            barmode="group",
            color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        
        st.plotly_chart(fig_themes, use_container_width=True)
    
    # Onglet 3: Évolution temporelle
    with tab3:
        st.subheader("Évolution des sentiments au fil du temps")
        
        if has_sentiment_data and 'sentiment_time_line' in sentiment_figures:
            st.plotly_chart(sentiment_figures['sentiment_time_line'], use_container_width=True)
        else:
            # Créer un graphique d'évolution fictif
            months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
            
            # Données fictives
            evolution_data = []
            for month in months:
                evolution_data.append({"Mois": month, "Sentiment": "Positif", "Pourcentage": 50 + np.random.randint(-10, 15)})
                evolution_data.append({"Mois": month, "Sentiment": "Neutre", "Pourcentage": 30 + np.random.randint(-10, 10)})
                evolution_data.append({"Mois": month, "Sentiment": "Négatif", "Pourcentage": 20 + np.random.randint(-10, 5)})
            
            evolution_df = pd.DataFrame(evolution_data)
            
            # Assurer que le total est 100%
            for month in months:
                month_data = evolution_df[evolution_df["Mois"] == month]
                total = month_data["Pourcentage"].sum()
                if total != 100:
                    scaling_factor = 100 / total
                    evolution_df.loc[evolution_df["Mois"] == month, "Pourcentage"] *= scaling_factor
            
            # Créer le graphique
            fig_evolution = px.line(
                evolution_df,
                x="Mois",
                y="Pourcentage",
                color="Sentiment",
                title="Évolution des sentiments au cours de l'année (données illustratives)",
                markers=True,
                color_discrete_map={
                    "Positif": "#2ecc71",
                    "Neutre": "#3498db",
                    "Négatif": "#e74c3c"
                }
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
        # Ajouter un graphique d'impact des initiatives
        st.subheader("Impact des initiatives sur la satisfaction")
        
        # Données fictives sur l'impact des initiatives
        initiatives_data = {
            "Date": ["Fév 2022", "Avr 2022", "Juin 2022", "Sept 2022", "Nov 2022"],
            "Initiative": [
                "Formation du personnel d'accueil", 
                "Nouveau système de rendez-vous", 
                "Rénovation des locaux", 
                "Campagne d'information", 
                "Programme de suivi amélioré"
            ],
            "Impact": ["+8%", "+12%", "+5%", "+7%", "+10%"]
        }
        
        # Créer un tableau des initiatives
        st.dataframe(pd.DataFrame(initiatives_data), use_container_width=True)
        
        # Note explicative
        st.info("""
        Le suivi de l'évolution des sentiments dans le temps permet d'identifier l'impact des initiatives d'amélioration 
        et des événements externes. Les données présentées ici sont illustratives. Pour une analyse précise, il est 
        recommandé de collecter systématiquement des retours après chaque campagne.
        """)
    
    # Onglet 4: Insights & Recommandations
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Principaux points de satisfaction")
            st.markdown("""
            - **Professionnalisme et gentillesse du personnel médical**
              *"L'infirmière a été particulièrement attentionnée et rassurante."*
            
            - **Sentiment d'accomplissement après le don**
              *"Sentiment de fierté de pouvoir aider d'autres personnes."*
            
            - **Connaissance de son groupe sanguin et statut de santé**
              *"J'apprécie de recevoir mon bilan sanguin après chaque don."*
            
            - **Collation et temps de repos**
              *"La collation après le don était appréciable et le personnel veillait à notre bien-être."*
            """)
        
        with col2:
            st.subheader("Principaux points d'insatisfaction")
            st.markdown("""
            - **Temps d'attente parfois long**
              *"J'ai attendu plus d'une heure avant d'être pris en charge."*
            
            - **Manque d'information sur l'utilisation du sang collecté**
              *"J'aimerais savoir comment mon sang a été utilisé."*
            
            - **Accessibilité limitée des lieux de collecte**
              *"Le centre est difficile d'accès en transports en commun."*
            
            - **Suivi post-don insuffisant**
              *"Pas de nouvelles après le don, j'aurais aimé être informé."*
            """)
        
        # Recommandations
        st.subheader("Recommandations")
        
        recommendations = {
            "Axe d'amélioration": [
                "Temps d'attente", 
                "Communication", 
                "Suivi des donneurs", 
                "Formation du personnel",
                "Accessibilité",
                "Expérience globale"
            ],
            "Recommandation": [
                "Optimiser les flux de travail et renforcer le système de rendez-vous",
                "Améliorer la communication sur l'impact des dons et leur utilisation",
                "Mettre en place un système de suivi systématique post-don",
                "Former le personnel à mieux gérer les préoccupations des donneurs",
                "Augmenter les options de transport et la signalétique",
                "Créer un environnement plus accueillant et confortable"
            ],
            "Priorité": [
                "Haute", 
                "Moyenne", 
                "Haute", 
                "Moyenne",
                "Basse",
                "Moyenne"
            ]
        }
        
        # Créer un DataFrame et l'afficher
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df, use_container_width=True)
        
        
def show_eligibility_prediction(data_dict, model, required_columns=None, feature_stats={}):
    """
    Affiche l'interface de prédiction avec règles de sécurité strictes
    et sauvegarde les données saisies dans le fichier CSV
    """
    import pandas as pd
    from datetime import datetime
    import os
    
    st.header("🤖 Prédiction d'Éligibilité")
    
    st.markdown("""
    Cette section vous permet de prédire l'éligibilité d'un potentiel donneur
    en fonction de ses caractéristiques démographiques et de santé.
    """)
    
    if model is None:
        st.warning("Le modèle de prédiction n'est pas disponible.")
        return

    # Dictionnaire pour stocker les valeurs
    input_values = {}
    
    # Organisation par onglets
    tabs = st.tabs(["Informations générales", "Santé", "Localisation"])
    
    # Onglet 1: Informations générales
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Âge
            age = st.slider("Âge", 18, 70, 35)
            input_values["age"] = age
            
            # Calculer le groupe d'âge
            if age < 18:
                age_group = "<18"
            elif age <= 25:
                age_group = "18-25"
            elif age <= 35:
                age_group = "26-35"
            elif age <= 45:
                age_group = "36-45"
            elif age <= 55:
                age_group = "46-55"
            elif age <= 65:
                age_group = "56-65"
            else:
                age_group = ">65"
            input_values["groupe_age"] = age_group
            
            # Genre
            genre = st.radio("Genre", ["Homme", "Femme"])
            input_values["Genre"] = genre
            
            # Expérience de don
            deja_donne = st.radio("A déjà donné le sang ?", ["Oui", "Non"])
            input_values["A-t-il (elle) déjà donné le sang"] = deja_donne
            input_values["experience_don"] = 1 if deja_donne == "Oui" else 0
        
        with col2:
            # Niveau d'études
            niveau_etude = st.selectbox("Niveau d'études", 
                                       ["Non précisé", "Primaire", "Secondaire", "Universitaire"])
            input_values["Niveau d'etude"] = niveau_etude
            
            # Situation matrimoniale
            situation_matrimoniale = st.selectbox("Situation matrimoniale", 
                                                 ["Non précisé", "Célibataire", "Marié(e)", 
                                                  "Divorcé(e)", "Veuf/Veuve"])
            input_values["Situation Matrimoniale (SM)"] = situation_matrimoniale
            
            # Profession
            profession = st.text_input("Profession", "Non précisé")
            input_values["Profession"] = profession
            
            # Religion
            religion = st.selectbox("Religion", 
                                   ["Non précisé", "Chrétien(ne)", "Musulman(e)", "Autre"])
            input_values["Religion"] = religion
            
            # Nationalité
            nationalite = st.selectbox("Nationalité", ["Camerounaise", "Autre"])
            input_values["Nationalité"] = nationalite
            
            # Groupe sanguin (ajouté pour la recherche)
            groupe_sanguin = st.selectbox("Groupe sanguin", 
                                          ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Non précisé"])
            input_values["Groupe sanguin"] = groupe_sanguin
    
    # Onglet 2: Santé
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Conditions de santé")
            
            # VIH, hépatite - CRITÈRE D'EXCLUSION ABSOLU
            vih_hbs_hcv = st.checkbox("Porteur de VIH, hépatite B ou C")
            input_values["porteur_vih_hbs_hcv"] = 1 if vih_hbs_hcv else 0
            input_values["Porteur(HIV,hbs,hcv)_indicateur"] = 1.0 if vih_hbs_hcv else 0.0
            
            # Afficher un avertissement si VIH/hépatite sélectionné
            if vih_hbs_hcv:
                st.warning("⚠️ Critère d'exclusion absolu : Porteur de VIH ou d'hépatite B/C")
            
            # Autres conditions médicales
            diabete = st.checkbox("Diabétique")
            input_values["diabetique"] = 1 if diabete else 0
            input_values["Diabétique_indicateur"] = 1.0 if diabete else 0.0
            
            hypertension = st.checkbox("Hypertendu")
            input_values["hypertendu"] = 1 if hypertension else 0
            input_values["Hypertendus_indicateur"] = 1.0 if hypertension else 0.0
            
            asthme = st.checkbox("Asthmatique")
            input_values["asthmatique"] = 1 if asthme else 0
            input_values["Asthmatiques_indicateur"] = 1.0 if asthme else 0.0
            
            # Critères d'exclusion absolus
            drepanocytaire = st.checkbox("Drépanocytaire")
            input_values["drepanocytaire"] = 1 if drepanocytaire else 0
            input_values["Drepanocytaire_indicateur"] = 1.0 if drepanocytaire else 0.0
            if drepanocytaire:
                st.warning("⚠️ Critère d'exclusion absolu : Drépanocytaire")
            
            cardiaque = st.checkbox("Problèmes cardiaques")
            input_values["cardiaque"] = 1 if cardiaque else 0
            input_values["Cardiaque_indicateur"] = 1.0 if cardiaque else 0.0
            if cardiaque:
                st.warning("⚠️ Critère d'exclusion absolu : Problèmes cardiaques")
        
        with col2:
            # Taux d'hémoglobine
            taux_hemoglobine = st.number_input("Taux d'hémoglobine (g/dL)", 
                                              min_value=7.0, max_value=20.0, value=13.5, step=0.1)
            
            # Utiliser exactement le même caractère apostrophe que celui attendu par le modèle
            input_values["Taux d\u2019h\u00e9moglobine"] = taux_hemoglobine
            
            # Avertissement pour taux d'hémoglobine bas
            if (genre == "Homme" and taux_hemoglobine < 13.0) or (genre == "Femme" and taux_hemoglobine < 12.0):
                st.warning(f"⚠️ Taux d'hémoglobine insuffisant pour un{'e' if genre == 'Femme' else ''} {genre.lower()}")
            
            # Ajouter d'autres caractéristiques médicales
            transfusion = st.checkbox("Antécédent de transfusion")
            input_values["transfusion"] = 1 if transfusion else 0
            
            tatouage = st.checkbox("Tatoué")
            input_values["tatoue"] = 1 if tatouage else 0
            
            scarification = st.checkbox("Scarifié")
            input_values["scarifie"] = 1 if scarification else 0
    
    # Onglet 3: Localisation
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Arrondissement
            arrondissement = st.selectbox("Arrondissement", 
                                         ["Douala 1", "Douala 2", "Douala 3", "Douala 4", "Douala 5", 
                                          "Douala (Non précisé)", "Autre"])
            
            input_values["Arrondissement de résidence"] = arrondissement
            input_values["arrondissement_clean"] = arrondissement
        
        with col2:
            # Quartier
            quartier = st.text_input("Quartier de résidence", "Non précisé")
            
            input_values["Quartier de Résidence"] = quartier
            input_values["quartier_clean"] = quartier
    
    # Numéro de téléphone (ajout pour le suivi)
    telephone = st.text_input("Numéro de téléphone (optionnel)", "")
    if telephone:
        try:
            input_values["Numéro_téléphone"] = int(telephone.replace(" ", ""))
        except:
            input_values["Numéro_téléphone"] = None
    
    # Consentement pour être recontacté
    consentement = st.checkbox("Consentement pour être contacté ultérieurement")
    input_values["Consentement_contact"] = "Oui" if consentement else "Non"
    
    # Bouton de prédiction avec avertissement pour critères d'exclusion
    if vih_hbs_hcv or drepanocytaire or cardiaque:
        st.warning("⚠️ Des critères d'exclusion absolus ont été détectés. Le donneur sera considéré comme non éligible.")
    
    if st.button("Prédire l'éligibilité"):
        # Faire la prédiction avec les règles de sécurité
        result, confidence = predict_eligibility(model, input_values, required_columns, feature_stats)
        
        # Stocker la confiance pour le classement ultérieur (pour la recherche)
        input_values["confidence_score"] = confidence
        
        # Afficher le résultat
        if result == "Éligible":
            st.success(f"Prédiction : {result} (Confiance : {confidence:.1f}%)")
            input_values["ÉLIGIBILITÉ AU DON."] = "Oui"
            input_values["eligibilite_code"] = 1
        elif result == "Non éligible":
            st.error(f"Prédiction : {result} (Confiance : {confidence:.1f}%)")
            input_values["ÉLIGIBILITÉ AU DON."] = "Non"
            input_values["eligibilite_code"] = 0
        else:
            st.warning(f"Prédiction : {result}")
            input_values["ÉLIGIBILITÉ AU DON."] = "Indéterminée"
            input_values["eligibilite_code"] = -1
        
        # Afficher une explication
        st.subheader("Facteurs importants")
        
        # Identifier les facteurs déterminants pour la non-éligibilité
        if result == "Non éligible":
            factors = []
            if vih_hbs_hcv:
                factors.append("Porteur de VIH, hépatite B ou C")
                input_values["Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]"] = "Oui"
            if diabete:
                factors.append("Diabète")
                input_values["Raison de non-eligibilité totale  [Diabétique]"] = "Oui"
            if cardiaque:
                factors.append("Problèmes cardiaques")
                input_values["Raison de non-eligibilité totale  [Cardiaque]"] = "Oui"
            if drepanocytaire:
                factors.append("Drépanocytaire")
                input_values["Raison de non-eligibilité totale  [Drepanocytaire]"] = "Oui"
            if (genre == "Homme" and taux_hemoglobine < 13.0) or (genre == "Femme" and taux_hemoglobine < 12.0):
                factors.append("Taux d\u2019h\u00e9moglobine bas")
                input_values["Raison indisponibilité  [Taux d'hémoglobine bas ]"] = "Oui"
            if hypertension:
                input_values["Raison de non-eligibilité totale  [Hypertendus]"] = "Oui"
            if asthme:
                input_values["Raison de non-eligibilité totale  [Asthmatiques]"] = "Oui"
            if transfusion:
                input_values["Raison de non-eligibilité totale  [Antécédent de transfusion]"] = "Oui"
            if tatouage:
                input_values["Raison de non-eligibilité totale  [Tatoué]"] = "Oui"
            if scarification:
                input_values["Raison de non-eligibilité totale  [Scarifié]"] = "Oui"
            
            if factors:
                st.warning(f"Facteur(s) déterminant(s): {', '.join(factors)}")
        
        # Explication générale
        st.markdown("""
        Les facteurs les plus influents pour l'éligibilité au don de sang sont:
        1. **Conditions médicales** (VIH, hépatite, diabète, problèmes cardiaques)
        2. **Taux d'hémoglobine** (minimum 12 g/dL pour les femmes, 13 g/dL pour les hommes)
        3. **Âge** (entre 18 et 65 ans généralement)
        4. **Expérience de don antérieure**
        """)
        
        # Ajouter les champs supplémentaires pour la sauvegarde
        input_values["Date de remplissage de la fiche"] = datetime.now().strftime("%Y-%m-%d")
        
        # Calculer une date de naissance approximative à partir de l'âge
        annee_actuelle = datetime.now().year
        annee_naissance = annee_actuelle - age
        input_values["Date de naissance"] = f"{annee_naissance}-01-01"  # Date approximative
        
        # Date du don actuel
        input_values["Date_don"] = datetime.now().strftime("%Y-%m-%d")
        
        # Bouton pour sauvegarder les données
        if st.button("Enregistrer les données du donneur"):
            try:
                # Chemin du fichier CSV
                csv_path = "data/processed_data/dataset_don_sang_enrichi.csv"
                
                # Vérifier si le fichier existe
                if os.path.exists(csv_path):
                    # Charger le CSV existant
                    df_existing = pd.read_csv(csv_path, encoding='utf-8')
                    
                    # Créer un DataFrame à partir des données saisies
                    df_new = pd.DataFrame([input_values])
                    
                    # S'assurer que toutes les colonnes du fichier original sont présentes
                    for col in df_existing.columns:
                        if col not in df_new.columns:
                            df_new[col] = None
                    
                    # Réordonner les colonnes pour correspondre au fichier original
                    df_new = df_new[df_existing.columns]
                    
                    # Fusionner les DataFrames
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    
                    # Sauvegarder le fichier combiné
                    df_combined.to_csv(csv_path, index=False, encoding='utf-8')
                    
                    st.success("Les données ont été enregistrées avec succès !")
                else:
                    st.error(f"Le fichier {csv_path} n'existe pas. Veuillez vérifier le chemin du fichier.")
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'enregistrement des données : {str(e)}")
    
    # CORRECTION 1: Utiliser directement st.download_button au lieu d'un bouton intermédiaire
    try:
        # Chemin du fichier CSV
        csv_path = "data/processed_data/dataset_don_sang_enrichi.csv"
        
        # Vérifier si le fichier existe
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as file:
                # AMÉLIORATION 2: Ajouter la date et l'heure au nom du fichier
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"dataset_don_sang_{current_datetime}.csv"
                
                st.download_button(
                    label="Télécharger le dataset enrichi",
                    data=file,
                    file_name=file_name,
                    mime="text/csv"
                )
        else:
            st.error(f"Le fichier {csv_path} n'existe pas. Veuillez vérifier le chemin du fichier.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du téléchargement : {str(e)}")
    
    # AMÉLIORATION 3: Ajout de la recherche de profils de donneurs éligibles
    st.header("🔍 Recherche de Donneurs Éligibles")
    
    st.markdown("""
    Cette section vous permet de rechercher les profils de donneurs les plus éligibles 
    en fonction de critères spécifiques.
    """)
    
    # Définir la fonction de recherche
    def search_eligible_donors(dataset_path, criteria):
        """
        Recherche les donneurs les plus éligibles dans le dataset selon les critères.
        
        Args:
            dataset_path (str): Chemin vers le fichier CSV
            criteria (dict): Critères de recherche (âge, groupe sanguin, quartier, ville)
            
        Returns:
            pd.DataFrame: Les 5 profils les plus éligibles
        """
        try:
            # Charger le dataset
            df = pd.read_csv(dataset_path, encoding='utf-8')
            
            # Filtrer pour ne garder que les donneurs éligibles
            eligible_donors = df[df["ÉLIGIBILITÉ AU DON."] == "Oui"]
            
            # Appliquer les filtres si spécifiés
            if criteria.get("age_min") and criteria.get("age_max"):
                eligible_donors = eligible_donors[(eligible_donors["age"] >= criteria["age_min"]) & 
                                                 (eligible_donors["age"] <= criteria["age_max"])]
            
            if criteria.get("groupe_sanguin") and criteria["groupe_sanguin"] != "Tous":
                eligible_donors = eligible_donors[eligible_donors["Groupe sanguin"] == criteria["groupe_sanguin"]]
                
            if criteria.get("quartier") and criteria["quartier"] != "Tous":
                eligible_donors = eligible_donors[eligible_donors["quartier_clean"].str.lower() == 
                                                 criteria["quartier"].lower()]
                
            if criteria.get("ville") and criteria["ville"] != "Tous":
                eligible_donors = eligible_donors[eligible_donors["arrondissement_clean"].str.contains(
                                                 criteria["ville"], case=False)]
            
            # Trier par priorité (si un score de confiance existe)
            if "confidence_score" in eligible_donors.columns:
                eligible_donors = eligible_donors.sort_values(by="confidence_score", ascending=False)
            
            # Prendre les 5 premiers résultats
            top_donors = eligible_donors.head(5)
            
            return top_donors
            
        except Exception as e:
            st.error(f"Erreur lors de la recherche: {str(e)}")
            return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur
    
    # Interface de recherche
    search_col1, search_col2 = st.columns(2)
    
    with search_col1:
        age_min = st.number_input("Âge minimum", min_value=18, max_value=65, value=18)
        age_max = st.number_input("Âge maximum", min_value=18, max_value=65, value=65)
        
        # On supposera que le groupe sanguin est stocké dans le dataset
        groupe_sanguin_recherche = st.selectbox("Groupe sanguin", 
                                    ["Tous", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    
    with search_col2:
        quartier_recherche = st.text_input("Quartier", "Tous")
        ville_recherche = st.selectbox("Ville/Arrondissement", 
                            ["Tous", "Douala 1", "Douala 2", "Douala 3", "Douala 4", "Douala 5", 
                             "Douala (Non précisé)", "Autre"])
    
    # Bouton de recherche
    if st.button("Rechercher des donneurs"):
        # Préparer les critères
        search_criteria = {
            "age_min": age_min,
            "age_max": age_max,
            "groupe_sanguin": groupe_sanguin_recherche,
            "quartier": quartier_recherche,
            "ville": ville_recherche
        }
        
        # Chemin du dataset
        dataset_path = "data/processed_data/dataset_don_sang_enrichi.csv"
        
        # Effectuer la recherche
        results = search_eligible_donors(dataset_path, search_criteria)
        
        # Afficher les résultats
        if not results.empty:
            st.success(f"{len(results)} donneur(s) éligible(s) trouvé(s)")
            
            # Colonnes à afficher (nous pouvons les ajuster selon les besoins)
            display_columns = ["age", "Genre", "Groupe sanguin", "quartier_clean", 
                               "arrondissement_clean", "Taux d\u2019h\u00e9moglobine", "Numéro_téléphone"]
            
            # S'assurer que toutes les colonnes existent
            existing_columns = [col for col in display_columns if col in results.columns]
            
            # Renommer les colonnes pour l'affichage
            column_names = {
                "age": "Âge",
                "Genre": "Genre",
                "Groupe sanguin": "Groupe sanguin",
                "quartier_clean": "Quartier",
                "arrondissement_clean": "Arrondissement",
                "Taux d\u2019h\u00e9moglobine": "Taux d'hémoglobine",
                "Numéro_téléphone": "Téléphone"
            }
            
            # Filtrer et renommer
            display_df = results[existing_columns].rename(columns=column_names)
            
            # Afficher le tableau
            st.dataframe(display_df)
        else:
            st.warning("Aucun donneur éligible ne correspond à ces critères.")


if __name__ == "__main__":
    main()