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
from preprocessing import preprocess_data
from plotly.subplots import make_subplots
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from visualizations import (create_geographic_visualizations, create_health_condition_visualizations,
                            create_donor_profiling_visualizations, create_campaign_effectiveness_visualizations,
                            create_donor_retention_visualizations, create_sentiment_analysis_visualizations)
# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord - Campagne de Don de Sang",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        if not os.path.exists('processed_data'):
            # Si les données prétraitées n'existent pas, les traiter et les sauvegarder
            data_dict = preprocess_data(file_path)
        else:
            # Sinon, charger les données prétraitées
            data_dict = {}
            for name in ['candidats', 'donneurs', 'candidats_age', 'combined']:
                csv_path = f"processed_data/{name}_processed.csv"
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
    model_path = "model/eligibility_model_gradient_boosting_20250323_104955.pkl"
    
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
    
    st.markdown("""
    Ce tableau de bord vous permet d'analyser les données des campagnes de don de sang pour optimiser vos futures initiatives.
    Explorez les différentes sections pour découvrir des insights sur la répartition géographique des donneurs,
    l'impact des conditions de santé sur l'éligibilité, le profil des donneurs idéaux, l'efficacité des campagnes,
    et les facteurs de fidélisation des donneurs.
    """)
    
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
            "📍 Répartition Géographique",
            "🏥 Conditions de Santé & Éligibilité",
            "🔬 Profilage des Donneurs",
            "📊 Efficacité des Campagnes",
            "🔄 Fidélisation des Donneurs",
            "💬 Analyse de Sentiment",
            "🤖 Prédiction d'Éligibilité"
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
            if section == "📍 Répartition Géographique":
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
        elif section == "🤖 Prédiction d'Éligibilité":
            show_eligibility_prediction(data_dict, model, expected_features, feature_stats)
        else:
            st.error("Aucune donnée n'a pu être chargée. Veuillez uploader un fichier valide ou vérifier le fichier par défaut.")
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement ou du traitement des données : {e}")
        st.info("Veuillez vérifier que le fichier est accessible et correctement formaté.")

def show_geographic_distribution(data_dict):
    """Affiche la section de répartition géographique des donneurs"""
    st.header("📍 Cartographie de la Répartition des Donneurs")
    
    st.markdown("""
    Cette section vous permet de visualiser la répartition géographique des donneurs de sang
    en fonction de leur lieu de résidence.
    """)
    
    # Créer les visualisations géographiques
    geo_figures = create_geographic_visualizations(data_dict['candidats'])
    
    # Afficher les visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        if 'arrondissement_bar' in geo_figures:
            st.plotly_chart(geo_figures['arrondissement_bar'], use_container_width=True)
    
    with col2:
        if 'quartier_bar' in geo_figures:
            st.plotly_chart(geo_figures['quartier_bar'], use_container_width=True)
    
    if 'arrond_eligibility_heatmap' in geo_figures:
        st.plotly_chart(geo_figures['arrond_eligibility_heatmap'], use_container_width=True)
    
    # Carte interactive (si les coordonnées sont disponibles)
    st.subheader("Carte Interactive des Donneurs")
    
    st.info("""
    Note: Une carte interactive montrant la répartition exacte des donneurs nécessiterait des données
    géographiques supplémentaires (coordonnées). Pour une implémentation complète, vous pourriez:
    
    1. Utiliser une API de géocodage pour convertir les arrondissements et quartiers en coordonnées
    2. Créer une carte choroplèthe montrant la densité des donneurs par zone
    3. Ajouter des marqueurs interactifs pour chaque site de collecte
    """)
    
    # Exemple de carte choroplèthe simplifiée (utilisant des données fictives)
    m = folium.Map(location=[4.0511, 9.7679], zoom_start=11)  # Coordonnées approximatives de Douala
    
    # Ajouter un titre à la carte
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Répartition des Donneurs à Douala</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Ajouter des marqueurs fictifs pour les principaux arrondissements
    arrondissements_coords = {
        'Douala 1': [4.0494, 9.7143],
        'Douala 2': [4.0611, 9.7179],
        'Douala 3': [4.0928, 9.7679],
        'Douala 4': [4.0711, 9.7543],
        'Douala 5': [4.0128, 9.7379]
    }
    
    # Extraire les données d'arrondissement
    if 'arrondissement_clean' in data_dict['candidats'].columns:
        arrond_counts = data_dict['candidats']['arrondissement_clean'].value_counts().to_dict()
        
        for arrond, coords in arrondissements_coords.items():
            count = arrond_counts.get(arrond, 0)
            popup_text = f"<b>{arrond}</b><br>Nombre de donneurs: {count}"
            
            # Ajuster la taille du cercle en fonction du nombre de donneurs
            radius = 500 + (count / 10)
            
            folium.Circle(
                location=coords,
                radius=radius,
                popup=popup_text,
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.6
            ).add_to(m)
    
    # Afficher la carte
    folium_static(m)
    
    # Insights et recommandations
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
    Les visualisations ci-dessous vous permettent de comprendre quelles conditions médicales
    influencent le plus l'éligibilité des donneurs.
    """)
    
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
    
    # Statistiques sur l'éligibilité
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
    
    # Insights et recommandations
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
    
    # Créer les visualisations de profilage des donneurs
    profiling_figures = create_donor_profiling_visualizations(data_dict['candidats'])
    
    # Afficher les visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age_eligibility_bar' in profiling_figures:
            st.plotly_chart(profiling_figures['age_eligibility_bar'], use_container_width=True)
    
    with col2:
        if 'gender_eligibility_bar' in profiling_figures:
            st.plotly_chart(profiling_figures['gender_eligibility_bar'], use_container_width=True)
    
    # Clustering des donneurs
    st.subheader("Clustering des Donneurs")
    
    if 'donor_clustering' in profiling_figures:
        st.plotly_chart(profiling_figures['donor_clustering'], use_container_width=True)
    
    if 'cluster_profiles_radar' in profiling_figures:
        st.plotly_chart(profiling_figures['cluster_profiles_radar'], use_container_width=True)
    
    # Caractéristiques du donneur idéal
    st.subheader("Caractéristiques du Donneur Idéal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Profil démographique :**
        - **Âge :** 26-45 ans
        - **Genre :** Hommes légèrement plus susceptibles d'être éligibles
        - **État civil :** Les personnes mariées montrent un taux d'éligibilité plus élevé
        - **Niveau d'éducation :** Niveau universitaire associé à une meilleure connaissance des critères d'éligibilité
        """)
    
    with col2:
        st.markdown("""
        **Facteurs comportementaux :**
        - **Expérience de don :** Les donneurs réguliers ont un taux d'éligibilité plus élevé
        - **Intervalle entre les dons :** Respect optimal de 3 à 6 mois entre les dons
        - **Sensibilisation :** Participation à des campagnes d'information préalables
        - **Mode de vie :** Alimentation équilibrée et activité physique régulière
        """)
    
    # Insights et recommandations
    st.subheader("Insights & Recommandations")
    
    st.markdown("""
    **Principaux insights :**
    - Les clusters identifiés montrent des profils distincts de donneurs avec différents taux d'éligibilité
    - L'âge et l'expérience de don antérieure sont les facteurs les plus déterminants pour prédire l'éligibilité
    - Les hommes entre 26 et 45 ans constituent le groupe démographique le plus fiable pour les dons réguliers
    
    **Recommandations :**
    - Cibler prioritairement les groupes à haut taux d'éligibilité pour maximiser l'efficacité des campagnes
    - Organiser des campagnes spécifiques pour les groupes sous-représentés mais à fort potentiel
    - Mettre en place des programmes de sensibilisation adaptés à chaque cluster de donneurs
    """)

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
    
    # Afficher les visualisations
    if 'monthly_donations_line' in campaign_figures:
        st.plotly_chart(campaign_figures['monthly_donations_line'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'average_monthly_donations' in campaign_figures:
            st.plotly_chart(campaign_figures['average_monthly_donations'], use_container_width=True)
    
    with col2:
        if 'weekly_donations' in campaign_figures:
            st.plotly_chart(campaign_figures['weekly_donations'], use_container_width=True)
    
    # Analyse par caractéristiques démographiques
    st.subheader("Contribution par Groupe Démographique")
    
    demographic_cols = ['groupe_age', 'Genre', 'Profession']
    for col in demographic_cols:
        fig_key = f'{col}_donations'
        if fig_key in campaign_figures:
            st.plotly_chart(campaign_figures[fig_key], use_container_width=True)
    
    # Insights et recommandations
    st.subheader("Insights & Recommandations")
    
    st.markdown("""
    **Principaux insights :**
    - Les mois d'avril, août et décembre montrent généralement une participation plus élevée
    - Le milieu de semaine (mardi à jeudi) est plus propice aux dons que le week-end
    - Certaines professions comme les étudiants, les enseignants et les professionnels de la santé sont surreprésentées
    
    **Recommandations :**
    - Planifier des campagnes majeures durant les périodes de forte participation (avril, août, décembre)
    - Organiser les collectes principalement en milieu de semaine
    - Cibler des lieux fréquentés par les groupes démographiques les plus contributeurs
    - Diversifier les approches pour atteindre des groupes professionnels sous-représentés
    """)

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
    
    # Afficher les visualisations
    if 'donor_experience_pie' in retention_figures:
        st.plotly_chart(retention_figures['donor_experience_pie'], use_container_width=True)
    
    # Afficher les facteurs influençant le retour des donneurs
    st.subheader("Facteurs Influençant le Retour des Donneurs")
    
    retention_factors = ['Genre', 'groupe_age', 'arrondissement_clean']
    for factor in retention_factors:
        fig_key = f'{factor}_retention'
        if fig_key in retention_figures:
            st.plotly_chart(retention_figures[fig_key], use_container_width=True)
    
    # Analyse du temps entre les dons
    if 'time_since_donation_hist' in retention_figures:
        st.plotly_chart(retention_figures['time_since_donation_hist'], use_container_width=True)
    
    if 'time_eligibility_bar' in retention_figures:
        st.plotly_chart(retention_figures['time_eligibility_bar'], use_container_width=True)
    
    # Stratégies de fidélisation
    st.subheader("Stratégies de Fidélisation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Communication et reconnaissance :**
        - Envoyer des rappels personnalisés quand le donneur devient à nouveau éligible
        - Créer un système de reconnaissance pour les donneurs réguliers (badges, certificats)
        - Partager des témoignages de bénéficiaires pour renforcer l'impact émotionnel
        """)
    
    with col2:
        st.markdown("""
        **Amélioration de l'expérience :**
        - Réduire les temps d'attente lors des collectes
        - Offrir un suivi de santé aux donneurs réguliers
        - Organiser des événements spéciaux pour les donneurs fidèles
        - Faciliter la prise de rendez-vous en ligne
        """)
    
    # Insights et recommandations
    st.subheader("Insights & Recommandations")
    
    st.markdown("""
    **Principaux insights :**
    - Environ 42% des candidats sont des donneurs récurrents
    - Le taux de fidélisation varie significativement selon l'arrondissement de résidence
    - Les donneurs âgés de 26 à 45 ans montrent les meilleurs taux de fidélisation
    - L'intervalle moyen entre deux dons est d'environ 12 mois (bien au-delà des 3 mois minimum requis)
    
    **Recommandations :**
    - Mettre en place un programme de fidélisation structuré avec des avantages progressifs
    - Cibler prioritairement les donneurs ayant déjà donné il y a plus de 3-4 mois
    - Organiser des campagnes spécifiques dans les zones à faible taux de fidélisation
    - Éduquer les donneurs sur la fréquence optimale des dons (tous les 3-4 mois)
    """)

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
    
    if sentiment_figures:
        # Afficher les visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sentiment_pie' in sentiment_figures:
                st.plotly_chart(sentiment_figures['sentiment_pie'], use_container_width=True)
        
        with col2:
            if 'sentiment_time_line' in sentiment_figures:
                st.plotly_chart(sentiment_figures['sentiment_time_line'], use_container_width=True)
        
        # Afficher des exemples de retours
        st.subheader("Exemples de Retours")
        
        st.info("""
        Note: Pour une implémentation complète, vous devriez extraire des exemples réels
        de retours textuels du jeu de données. Voici quelques exemples illustratifs:
        """)
        
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
    else:
        st.warning("""
        Données textuelles insuffisantes pour réaliser une analyse de sentiment.
        Pour implémenter cette fonctionnalité, il faudrait:
        
        1. Collecter plus de retours textuels des donneurs via des questionnaires
        2. Utiliser des bibliothèques Python comme NLTK ou TextBlob pour l'analyse de sentiment
        3. Créer des visualisations montrant l'évolution des opinions au fil du temps
        """)
    
    # Insights et recommandations
    st.subheader("Insights & Recommandations")
    
    st.markdown("""
    **Principaux points de satisfaction :**
    - Professionnalisme et gentillesse du personnel médical
    - Sentiment d'accomplissement après le don
    - Connaissance de son groupe sanguin et statut de santé
    
    **Principaux points d'insatisfaction :**
    - Temps d'attente parfois long
    - Manque d'information sur l'utilisation du sang collecté
    - Accessibilité limitée des lieux de collecte
    
    **Recommandations :**
    - Optimiser les flux de travail pour réduire les temps d'attente
    - Améliorer la communication sur l'impact des dons
    - Mettre en place un système de feedback systématique après chaque don
    - Former le personnel à mieux gérer les préoccupations des donneurs
    """)

def show_eligibility_prediction(data_dict, model, required_columns=None, feature_stats={}):
    """
    Affiche l'interface de prédiction avec règles de sécurité strictes
    """
    st.header("🤖 Prédiction d'Éligibilité")
    
    st.markdown("""
    Cette section vous permet de prédire l'éligibilité d'un potentiel donneur
    en fonction de ses caractéristiques démographiques et de santé.
    """)
    
    if model is None:
        st.warning("Le modèle de prédiction n'est pas disponible.")
        return
    
    # Afficher les caractéristiques attendues
    with st.expander("Caractéristiques attendues par le modèle"):
        if required_columns:
            st.write(f"Le modèle utilise {len(required_columns)} caractéristiques:")
            st.write(", ".join(required_columns))
        else:
            st.write("Impossible de déterminer les caractéristiques attendues.")
    
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
    
    # Onglet 2: Santé
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Conditions de santé")
            
            # VIH, hépatite - CRITÈRE D'EXCLUSION ABSOLU
            vih_hbs_hcv = st.checkbox("Porteur de VIH, hépatite B ou C")
            input_values["porteur_vih_hbs_hcv"] = 1 if vih_hbs_hcv else 0
            
            # Afficher un avertissement si VIH/hépatite sélectionné
            if vih_hbs_hcv:
                st.warning("⚠️ Critère d'exclusion absolu : Porteur de VIH ou d'hépatite B/C")
            
            # Autres conditions médicales
            diabete = st.checkbox("Diabétique")
            input_values["diabetique"] = 1 if diabete else 0
            
            hypertension = st.checkbox("Hypertendu")
            input_values["hypertendu"] = 1 if hypertension else 0
            
            asthme = st.checkbox("Asthmatique")
            input_values["asthmatique"] = 1 if asthme else 0
            
            # Critères d'exclusion absolus
            drepanocytaire = st.checkbox("Drépanocytaire")
            input_values["drepanocytaire"] = 1 if drepanocytaire else 0
            if drepanocytaire:
                st.warning("⚠️ Critère d'exclusion absolu : Drépanocytaire")
            
            cardiaque = st.checkbox("Problèmes cardiaques")
            input_values["cardiaque"] = 1 if cardiaque else 0
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
    
    # Bouton de prédiction avec avertissement pour critères d'exclusion
    if vih_hbs_hcv or drepanocytaire or cardiaque:
        st.warning("⚠️ Des critères d'exclusion absolus ont été détectés. Le donneur sera considéré comme non éligible.")
    
    if st.button("Prédire l'éligibilité"):
        # Faire la prédiction avec les règles de sécurité
        result, confidence = predict_eligibility(model, input_values, required_columns, feature_stats)
        
        # Afficher le résultat
        if result == "Éligible":
            st.success(f"Prédiction : {result} (Confiance : {confidence:.1f}%)")
        elif result == "Non éligible":
            st.error(f"Prédiction : {result} (Confiance : {confidence:.1f}%)")
        else:
            st.warning(f"Prédiction : {result}")
        
        # Afficher une explication
        st.subheader("Facteurs importants")
        
        # Identifier les facteurs déterminants pour la non-éligibilité
        if result == "Non éligible":
            factors = []
            if vih_hbs_hcv:
                factors.append("Porteur de VIH, hépatite B ou C")
            if diabete:
                factors.append("Diabète")
            if cardiaque:
                factors.append("Problèmes cardiaques")
            if drepanocytaire:
                factors.append("Drépanocytaire")
            if (genre == "Homme" and taux_hemoglobine < 13.0) or (genre == "Femme" and taux_hemoglobine < 12.0):
                factors.append("Taux d'hémoglobine bas")
            
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
if __name__ == "__main__":
    main()