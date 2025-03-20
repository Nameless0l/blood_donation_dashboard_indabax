import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from streamlit_folium import folium_static
import folium
import os
import sys

# Ajout du chemin pour l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des modules personnalisés
from src.data_processing import load_and_process_data, prepare_geo_data
from src.visualization import (create_donor_map, health_conditions_chart, eligibility_by_condition,
                            donor_clustering, campaign_effectiveness, donor_retention_analysis,
                            sentiment_analysis)
from src.ml_models import train_eligibility_model, predict_eligibility
from src.utils import format_number, create_kpi_card

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Tableau de bord de don de sang - Douala",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Titre et introduction
    st.title("Tableau de bord d'analyse des campagnes de don de sang à Douala")
    st.markdown("""
    Ce tableau de bord présente une analyse complète des données de campagnes de don de sang à Douala.
    Explorez les différentes sections pour obtenir des insights sur la distribution géographique
    des donneurs, les conditions de santé, les profils de donateurs idéaux et bien plus encore.
    """)
    
    # Uploader un fichier ou utiliser le fichier par défaut
    uploaded_file = st.sidebar.file_uploader("Charger un fichier de données", type=["csv", "xlsx", "xls"])
    
    # Chemin par défaut pour votre dataset
    # default_path = "data/Updated Challenge dataset.xlsx"
    default_path = "data/Updated Challenge dataset.xlsx"
    
    # Charger les données
    with st.spinner("Chargement des données..."):
        if uploaded_file is not None:
            # Sauvegarder temporairement le fichier
            temp_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                df = load_and_process_data(temp_path)
                st.success(f"Données chargées avec succès! {len(df)} enregistrements trouvés.")
            except Exception as e:
                st.error(f"Erreur lors du chargement des données: {str(e)}")
                st.info("Génération de données factices pour démonstration...")
                from src.data_processing import generate_dummy_data
                df = generate_dummy_data(500)
        else:
            # Vérifier si le fichier par défaut existe
            if os.path.exists(default_path):
                try:
                    df = load_and_process_data(default_path)
                    st.success(f"Données chargées avec succès! {len(df)} enregistrements trouvés.")
                except Exception as e:
                    st.error(f"Erreur lors du chargement des données par défaut: {str(e)}")
                    st.info("Génération de données factices pour démonstration...")
                    from src.data_processing import generate_dummy_data
                    df = generate_dummy_data(500)
            else:
                st.info("Fichier de données par défaut non trouvé. Génération de données factices pour démonstration...")
                from src.data_processing import generate_dummy_data
                df = generate_dummy_data(500)
    
    # Barre de navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", [
        "Carte de répartition des donneurs", 
        "Conditions de santé et éligibilité",
        "Profilage des donateurs idéaux",
        "Efficacité des campagnes",
        "Fidélisation des donateurs",
        "Analyse des sentiments",
        "Prédiction d'éligibilité"
    ])
    
    # Filtres généraux dans la barre latérale
    st.sidebar.title("Filtres")
    
    # Filtre par arrondissement
    all_arrondissements = ["Tous"] + sorted(df['arrondissement'].unique().tolist())
    selected_arrondissement = st.sidebar.selectbox("Arrondissement", all_arrondissements)
    
    # Filtre par tranche d'âge
    age_min, age_max = int(df['age'].min()), int(df['age'].max())
    age_range = st.sidebar.slider("Tranche d'âge", age_min, age_max, (age_min, age_max))
    
    # Filtre par sexe
    all_genders = ["Tous"] + sorted(df['sexe'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Sexe", all_genders)
    
    # Appliquer les filtres
    filtered_df = df.copy()
    
    if selected_arrondissement != "Tous":
        filtered_df = filtered_df[filtered_df['arrondissement'] == selected_arrondissement]
    
    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    if selected_gender != "Tous":
        filtered_df = filtered_df[filtered_df['sexe'] == selected_gender]
    
    # Afficher le nombre de donneurs filtrés
    st.sidebar.info(f"Nombre de donneurs après filtrage: {len(filtered_df)}")
    
    # Contenu principal basé sur la page sélectionnée
    if page == "Carte de répartition des donneurs":
        display_donor_map(filtered_df)
    elif page == "Conditions de santé et éligibilité":
        display_health_conditions(filtered_df)
    elif page == "Profilage des donateurs idéaux":
        display_donor_clustering(filtered_df)
    elif page == "Efficacité des campagnes":
        display_campaign_effectiveness(filtered_df)
    elif page == "Fidélisation des donateurs":
        display_donor_retention(filtered_df)
    elif page == "Analyse des sentiments":
        display_sentiment_analysis(filtered_df)
    elif page == "Prédiction d'éligibilité":
        display_eligibility_prediction(df)

def display_donor_map(df):
    st.header("Carte de répartition des donneurs")
    st.markdown("Visualisation de la distribution géographique des donneurs de sang par arrondissement et quartier.")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_donors = len(df)
        st.metric("Total des donneurs", format_number(total_donors))
    
    with col2:
        eligible_donors = df['eligible'].sum()
        eligible_pct = eligible_donors / total_donors if total_donors > 0 else 0
        st.metric("Donneurs éligibles", format_number(eligible_donors), f"{eligible_pct:.1%}")
    
    with col3:
        districts = df['arrondissement'].nunique()
        st.metric("Arrondissements", districts)
    
    with col4:
        neighborhoods = df['quartier'].nunique()
        st.metric("Quartiers", neighborhoods)
    
    # Carte
    with st.spinner("Génération de la carte..."):
        try:
            donor_map = create_donor_map(df)
            folium_static(donor_map)
        except Exception as e:
            st.error(f"Erreur lors de la génération de la carte: {str(e)}")
            st.info("La visualisation de la carte nécessite des données géographiques valides.")
    
    # Statistiques par arrondissement
    st.subheader("Statistiques par arrondissement")
    arrond_stats = df.groupby('arrondissement').agg({
        'id_donneur': 'count',
        'age': 'mean',
        'eligible': 'mean'
    }).reset_index()
    
    arrond_stats.columns = ['Arrondissement', 'Nombre de donneurs', 'Âge moyen', 'Taux d\'éligibilité']
    arrond_stats['Âge moyen'] = arrond_stats['Âge moyen'].round(1)
    arrond_stats['Taux d\'éligibilité'] = (arrond_stats['Taux d\'éligibilité'] * 100).round(1)
    arrond_stats = arrond_stats.sort_values('Nombre de donneurs', ascending=False)
    
    st.dataframe(arrond_stats)

def display_health_conditions(df):
    st.header("Conditions de santé et éligibilité")
    st.markdown("Analyse de l'impact des problèmes de santé sur l'admissibilité au don de sang.")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            health_fig = health_conditions_chart(df)
            st.plotly_chart(health_fig, use_container_width=True)
        
        with col2:
            elig_fig = eligibility_by_condition(df)
            st.plotly_chart(elig_fig, use_container_width=True)
        
        # Tableau des statistiques d'éligibilité
        st.subheader("Taux d'éligibilité par condition de santé")
        elig_stats = df.groupby('condition_sante').agg({
            'id_donneur': 'count',
            'eligible': 'mean'
        }).reset_index()
        elig_stats.columns = ['Condition de santé', 'Nombre de donneurs', 'Taux d\'éligibilité']
        elig_stats['Taux d\'éligibilité'] = (elig_stats['Taux d\'éligibilité'] * 100).round(1)
        elig_stats = elig_stats.sort_values('Taux d\'éligibilité', ascending=False)
        
        st.dataframe(elig_stats)
    except Exception as e:
        st.error(f"Erreur lors de la génération des visualisations: {str(e)}")

# Ajouter les autres fonctions d'affichage ici (similaires à celles dans l'application adaptée)
def display_donor_clustering(df):
    st.header("Profilage des donateurs idéaux")
    st.markdown("Regroupement des donneurs en profils similaires en fonction de caractéristiques démographiques et de santé.")
    
    try:
        with st.spinner("Calcul des clusters..."):
            fig, cluster_profiles = donor_clustering(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les profils de clusters
        st.subheader("Caractéristiques des groupes de donneurs identifiés")
        for i, profile in enumerate(cluster_profiles):
            with st.expander(f"Groupe {i+1} - {profile['count']} donneurs"):
                st.write(f"**Âge moyen:** {profile['age_mean']:.1f} ans")
                st.write(f"**Profession dominante:** {profile['top_profession']}")
                st.write(f"**Condition de santé dominante:** {profile['top_health_condition']}")
                st.write("**Répartition par sexe:**")
                for gender, ratio in profile['gender_ratio'].items():
                    st.write(f"- {gender}: {ratio:.1%}")
    except Exception as e:
        st.error(f"Erreur lors du clustering: {str(e)}")
        st.info("Le clustering nécessite des données numériques et catégorielles valides.")

def display_campaign_effectiveness(df):
    st.header("Efficacité des campagnes")
    st.markdown("Analyse des tendances temporelles et des facteurs influençant le succès des campagnes de don de sang.")
    
    try:
        # Visualisation de l'évolution temporelle
        fig_time, fig_month, fig_demo = campaign_effectiveness(df)
        
        st.subheader("Évolution des dons au fil du temps")
        st.plotly_chart(fig_time, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Saisonnalité des dons")
            st.plotly_chart(fig_month, use_container_width=True)
        
        with col2:
            st.subheader("Top professions des donneurs")
            st.plotly_chart(fig_demo, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des campagnes: {str(e)}")

def display_donor_retention(df):
    st.header("Fidélisation des donateurs")
    st.markdown("Analyse de la fidélité des donneurs et des facteurs démographiques associés aux dons répétés.")
    
    try:
        fig_loyalty, fig_age, fig_prof = donor_retention_analysis(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des donneurs par fidélité")
            st.plotly_chart(fig_loyalty, use_container_width=True)
        
        with col2:
            st.subheader("Fidélité selon l'âge")
            st.plotly_chart(fig_age, use_container_width=True)
        
        st.subheader("Top 10 des professions par fidélité")
        st.plotly_chart(fig_prof, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse de fidélisation: {str(e)}")

def display_sentiment_analysis(df):
    st.header("Analyse des sentiments")
    st.markdown("Analyse des commentaires et retours des donneurs pour identifier les tendances et sentiments.")
    
    try:
        fig_dist, wordcloud_figs = sentiment_analysis(df)
        
        st.subheader("Distribution des sentiments")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Afficher les wordclouds
        st.subheader("Mots-clés par sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (sentiment, fig) in enumerate(wordcloud_figs.items()):
            with [col1, col2, col3][i % 3]:
                st.write(f"**{sentiment}**")
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des sentiments: {str(e)}")
        st.info("L'analyse des sentiments nécessite des données textuelles.")

def display_eligibility_prediction(df):
    st.header("Prédiction d'éligibilité au don de sang")
    st.markdown("""
    Utilisez ce formulaire pour prédire si une personne est éligible au don de sang
    en fonction de ses caractéristiques personnelles et de santé.
    """)
    
    # Entraîner ou charger le modèle
    with st.spinner("Chargement du modèle..."):
        try:
            model = train_eligibility_model(df)
            st.success("Modèle chargé avec succès!")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {str(e)}")
            return
    
    # Formulaire de prédiction
    st.subheader("Entrez les informations du donneur potentiel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        sexe = st.selectbox("Sexe", df['sexe'].unique())
    
    with col2:
        profession = st.selectbox("Profession", df['profession'].unique())
        condition_sante = st.selectbox("Condition de santé", df['condition_sante'].unique())
    
    # Bouton de prédiction
    if st.button("Prédire l'éligibilité"):
        # Préparer les données
        input_data = {
            "age": age,
            "sexe": sexe,
            "profession": profession,
            "condition_sante": condition_sante
        }
        
        # Faire la prédiction
        try:
            # Essayer d'utiliser l'API si disponible
            try:
                response = requests.post("http://localhost:8000/predict/", json=input_data, timeout=2)
                result = response.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Si l'API n'est pas disponible, utiliser le modèle directement
                st.info("API non disponible. Utilisation du modèle local...")
                result = predict_eligibility(model, input_data)
            
            # Afficher le résultat
            if result["eligible"]:
                st.success(f"✅ Éligible au don de sang (Probabilité: {result['probability']:.2f})")
            else:
                st.error(f"❌ Non éligible au don de sang (Probabilité: {result['probability']:.2f})")
            
            st.info("Note: Cette prédiction est basée sur un modèle d'apprentissage automatique et ne remplace pas l'avis d'un professionnel de santé.")
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    main()