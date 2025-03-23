import io 
import os
import pickle
import folium
import tempfile
import base64
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
    Entraîne un modèle de prédiction d'éligibilité au don de sang
    """
    # Vérifier si le modèle existe déjà
    model_path = "model/eligibility_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # Créer le répertoire model s'il n'existe pas
    os.makedirs("model", exist_ok=True)
    
    # Préparer les données pour l'entraînement
    # Sélectionner les caractéristiques pertinentes
    features = []
    
    # Caractéristiques démographiques
    if 'age' in df.columns:
        features.append('age')
    
    if 'Genre' in df.columns:
        # Encoder le genre
        df['genre_code'] = df['Genre'].map({'Homme': 1, 'Femme': 0})
        features.append('genre_code')
    
    # Expérience de don antérieure
    if 'experience_don' in df.columns:
        features.append('experience_don')
    
    # Conditions de santé (indicateurs)
    health_indicators = [col for col in df.columns if '_indicateur' in col]
    features.extend(health_indicators)
    
    # Filtrer les lignes sans valeurs manquantes pour les caractéristiques sélectionnées
    model_df = df[features + ['eligibilite_code']].dropna()
    
    # Vérifier si nous avons suffisamment de données
    if len(model_df) < 20:
        return None
    
    # Diviser en caractéristiques et cible
    X = model_df[features]
    y = model_df['eligibilite_code']
    
    # Convertir y en catégories (éligible / non éligible)
    y_binary = (y == 1).astype(int)  # 1 pour éligible, 0 pour non éligible
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    # Entraîner un modèle RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Sauvegarder le modèle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def predict_eligibility(model, input_features):
    """
    Prédit l'éligibilité au don de sang à partir des caractéristiques d'entrée
    """
    if model is None:
        return "Modèle non disponible"
    
    # Faire la prédiction
    prediction = model.predict([input_features])[0]
    prediction_proba = model.predict_proba([input_features])[0]
    
    if prediction == 1:
        result = "Éligible"
        confidence = prediction_proba[1] * 100
    else:
        result = "Non éligible"
        confidence = prediction_proba[0] * 100
    
    return result, confidence

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
            model = train_eligibility_model(data_dict['candidats'])
            
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

def show_eligibility_prediction(data_dict, model):
    """Affiche la section de prédiction d'éligibilité"""
    st.header("🤖 Modèle de Prédiction d'Éligibilité")
    
    st.markdown("""
    Cette section vous permet de prédire l'éligibilité d'un potentiel donneur
    en fonction de ses caractéristiques démographiques et de santé. Cet outil
    peut être utilisé pour le pré-screening avant les campagnes.
    """)
    
    if model is None:
        st.warning("""
        Le modèle de prédiction n'a pas pu être créé en raison de données insuffisantes.
        Pour implémenter cette fonctionnalité, il faudrait:
        
        1. Collecter plus de données sur les donneurs et leur éligibilité
        2. Entraîner un modèle d'apprentissage automatique avec ces données
        3. Déployer le modèle via une API pour l'utiliser en temps réel
        """)
        return
    
    # Interface de prédiction
    st.subheader("Prédiction d'Éligibilité")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Caractéristiques démographiques
        st.markdown("**Caractéristiques démographiques**")
        
        age = st.slider("Âge", 18, 70, 35)
        genre = st.radio("Genre", ["Homme", "Femme"])
        already_donated = st.radio("A déjà donné du sang ?", ["Oui", "Non"])
        
        # Convertir en format numérique
        genre_code = 1 if genre == "Homme" else 0
        experience_don = 1 if already_donated == "Oui" else 0
    
    with col2:
        # Conditions de santé
        st.markdown("**Conditions de santé**")
        
        hiv_hbs_hcv = st.checkbox("Porteur de VIH, hépatite B ou C")
        diabete = st.checkbox("Diabétique")
        hypertension = st.checkbox("Hypertendu")
        asthme = st.checkbox("Asthmatique")
        drepano = st.checkbox("Drépanocytaire")
        cardiaque = st.checkbox("Problèmes cardiaques")
    
    # Préparer les caractéristiques d'entrée pour le modèle
    input_features = []
    
    # L'ordre des caractéristiques doit correspondre à celui utilisé lors de l'entraînement
    if 'age' in data_dict['candidats'].columns:
        input_features.append(age)
    
    if 'Genre' in data_dict['candidats'].columns:
        input_features.append(genre_code)
    
    if 'experience_don' in data_dict['candidats'].columns:
        input_features.append(experience_don)
    
    # Conditions de santé
    health_conditions = {
        "Porteur(HIV,hbs,hcv)_indicateur": 1 if hiv_hbs_hcv else 0,
        "Diabétique_indicateur": 1 if diabete else 0,
        "Hypertendus_indicateur": 1 if hypertension else 0,
        "Asthmatiques_indicateur": 1 if asthme else 0,
        "Drepanocytaire_indicateur": 1 if drepano else 0,
        "Cardiaque_indicateur": 1 if cardiaque else 0
    }
    
    # Ajouter les conditions de santé présentes dans les données d'entraînement
    for condition in ["Porteur(HIV,hbs,hcv)_indicateur", "Diabétique_indicateur", 
                      "Hypertendus_indicateur", "Asthmatiques_indicateur",
                      "Drepanocytaire_indicateur", "Cardiaque_indicateur"]:
        if condition in data_dict['candidats'].columns:
            input_features.append(health_conditions[condition])
    
    # Bouton de prédiction
    if st.button("Prédire l'éligibilité"):
        # Faire la prédiction
        result, confidence = predict_eligibility(model, input_features)
        
        # Afficher le résultat
        if result == "Éligible":
            st.success(f"Prédiction : {result} (Confiance : {confidence:.1f}%)")
        else:
            st.error(f"Prédiction : {result} (Confiance : {confidence:.1f}%)")
        
        # Afficher une explication
        st.subheader("Explication de la prédiction")
        
        # Obtenir l'importance des caractéristiques
        feature_names = []
        if 'age' in data_dict['candidats'].columns:
            feature_names.append("Âge")
        
        if 'Genre' in data_dict['candidats'].columns:
            feature_names.append("Genre")
        
        if 'experience_don' in data_dict['candidats'].columns:
            feature_names.append("Expérience de don")
        
        for condition in ["Porteur(HIV,hbs,hcv)_indicateur", "Diabétique_indicateur", 
                          "Hypertendus_indicateur", "Asthmatiques_indicateur",
                          "Drepanocytaire_indicateur", "Cardiaque_indicateur"]:
            if condition in data_dict['candidats'].columns:
                feature_names.append(condition.replace("_indicateur", ""))
        
        feature_importance = get_feature_importance(model, feature_names)
        
        if feature_importance is not None:
            # Créer un graphique d'importance des caractéristiques
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Importance des caractéristiques dans la prédiction",
                color='Importance',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Fournir une explication textuelle
            top_factors = feature_importance.head(3)['Feature'].tolist()
            
            st.markdown(f"""
            Les facteurs les plus influents dans cette prédiction sont: **{", ".join(top_factors)}**.
            
            {"Pour augmenter vos chances d'éligibilité, consultez un professionnel de santé pour discuter des facteurs de risque identifiés." if result == "Non éligible" else "Votre profil correspond à celui d'un donneur éligible typique."}
            """)
    
    # API Documentation
    st.subheader("Documentation de l'API")
    
    st.markdown("""
    Ce modèle de prédiction peut être intégré à votre site web ou application via une API REST.
    
    **Endpoint:** `/api/predict_eligibility`
    
    **Méthode:** POST
    
    **Format de requête:**
    ```json
    {
        "age": 35,
        "genre": "Homme",
        "experience_don": 1,
        "conditions_sante": {
            "vih_hbs_hcv": 0,
            "diabete": 0,
            "hypertension": 0,
            "asthme": 0,
            "drepanocytaire": 0,
            "cardiaque": 0
        }
    }
    ```
    
    **Format de réponse:**
    ```json
    {
        "prediction": "Éligible",
        "confidence": 92.5,
        "facteurs_importants": ["Âge", "Expérience de don", "Genre"]
    }
    ```
    
    Pour implémenter cette API, vous pouvez utiliser Flask ou FastAPI en Python.
    """)

if __name__ == "__main__":
    main()