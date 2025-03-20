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
from src.data_processing import load_and_process_data, prepare_geo_data, generate_dummy_data
from src.visualization import (create_donor_map, health_conditions_chart, eligibility_by_condition,
                            donor_clustering, campaign_effectiveness, donor_retention_analysis,
                            sentiment_analysis)
from src.ml_models import train_eligibility_model, predict_eligibility
from src.utils import format_number, create_kpi_card

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Tableau de bord de don de sang",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Titre et introduction
    st.title("Tableau de bord d'analyse des campagnes de don de sang")
    st.markdown("""
    Ce tableau de bord présente une analyse complète des données de campagnes de don de sang.
    Explorez les différentes sections pour obtenir des insights sur la distribution géographique
    des donneurs, les conditions de santé, les profils de donateurs idéaux et bien plus encore.
    """)
    
    # Uploader un fichier ou utiliser le fichier par défaut
    uploaded_file = st.sidebar.file_uploader("Charger un fichier de données", type=["csv", "xlsx", "xls"])
    
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
                df = generate_dummy_data(500)
        else:
            # Vérifier si un fichier de données existe dans le dossier data
            default_path = "data/blood_donation.xlsx"
            if os.path.exists(default_path):
                try:
                    df = load_and_process_data(default_path)
                    st.success(f"Données chargées avec succès! {len(df)} enregistrements trouvés.")
                except Exception as e:
                    st.error(f"Erreur lors du chargement des données: {str(e)}")
                    st.info("Génération de données factices pour démonstration...")
                    df = generate_dummy_data(500)
            else:
                st.info("Aucun fichier de données trouvé. Génération de données factices pour démonstration...")
                df = generate_dummy_data(500)
    
    # Afficher quelques statistiques générales
    st.subheader("Aperçu des données")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre total de donneurs", format_number(len(df)))
    
    with col2:
        eligible_count = df['eligible'].sum()
        st.metric("Donneurs éligibles", f"{eligible_count} ({eligible_count/len(df):.1%})")
    
    with col3:
        if 'arrondissement' in df.columns:
            st.metric("Nombre d'arrondissements", df['arrondissement'].nunique())
        else:
            st.metric("Nombre d'arrondissements", "N/A")
    
    with col4:
        if 'condition_sante' in df.columns:
            healthy_count = (df['condition_sante'] == 'Aucune').sum()
            st.metric("Donneurs sans condition médicale", f"{healthy_count} ({healthy_count/len(df):.1%})")
        else:
            st.metric("Donneurs sans condition médicale", "N/A")
    
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
    if 'arrondissement' in df.columns:
        all_arrondissements = ["Tous"] + sorted(df['arrondissement'].unique().tolist())
        selected_arrondissement = st.sidebar.selectbox("Arrondissement", all_arrondissements)
    else:
        selected_arrondissement = "Tous"
    
    # Filtre par tranche d'âge
    if 'age' in df.columns:
        age_min, age_max = int(df['age'].min()), int(df['age'].max())
        age_range = st.sidebar.slider("Tranche d'âge", age_min, age_max, (age_min, age_max))
    else:
        age_range = (18, 70)
    
    # Filtre par sexe
    if 'sexe' in df.columns:
        all_genders = ["Tous"] + sorted(df['sexe'].unique().tolist())
        selected_gender = st.sidebar.selectbox("Sexe", all_genders)
    else:
        selected_gender = "Tous"
    
    # Filtre par condition de santé
    if 'condition_sante' in df.columns:
        all_conditions = ["Toutes"] + sorted(df['condition_sante'].unique().tolist())
        selected_condition = st.sidebar.selectbox("Condition de santé", all_conditions)
    else:
        selected_condition = "Toutes"
    
    # Appliquer les filtres
    filtered_df = df.copy()
    
    if selected_arrondissement != "Tous":
        filtered_df = filtered_df[filtered_df['arrondissement'] == selected_arrondissement]
    
    if 'age' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    if selected_gender != "Tous":
        filtered_df = filtered_df[filtered_df['sexe'] == selected_gender]
    
    if selected_condition != "Toutes" and 'condition_sante' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['condition_sante'] == selected_condition]
    
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
        'age': 'mean' if 'age' in df.columns else lambda x: 0,
        'eligible': 'mean'
    }).reset_index()
    
    arrond_stats.columns = ['Arrondissement', 'Nombre de donneurs', 'Âge moyen', 'Taux d\'éligibilité']
    arrond_stats['Âge moyen'] = arrond_stats['Âge moyen'].round(1)
    arrond_stats['Taux d\'éligibilité'] = (arrond_stats['Taux d\'éligibilité'] * 100).round(1)
    arrond_stats = arrond_stats.sort_values('Nombre de donneurs', ascending=False)
    
    st.dataframe(arrond_stats)
    
    # Ajout d'un graphique montrant la distribution par arrondissement
    st.subheader("Distribution des donneurs par arrondissement")
    district_fig = px.bar(
        arrond_stats, 
        x='Arrondissement', 
        y='Nombre de donneurs',
        color='Taux d\'éligibilité',
        color_continuous_scale='Reds',
        labels={'Nombre de donneurs': 'Nombre de donneurs', 'Arrondissement': 'Arrondissement'},
        title='Distribution des donneurs par arrondissement'
    )
    st.plotly_chart(district_fig, use_container_width=True)

def display_health_conditions(df):
    st.header("Conditions de santé et éligibilité")
    st.markdown("Analyse de l'impact des problèmes de santé sur l'admissibilité au don de sang.")
    
    try:
        # Vérifier si la colonne condition_sante existe
        if 'condition_sante' not in df.columns:
            st.warning("La colonne 'condition_sante' n'existe pas dans les données. Certaines visualisations peuvent ne pas être disponibles.")
            return
            
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
        
        # Ajouter un graphique de distribution par âge et condition de santé
        if 'age' in df.columns and 'tranche_age' in df.columns:
            st.subheader("Distribution des conditions de santé par tranche d'âge")
            age_health_counts = df.groupby(['tranche_age', 'condition_sante']).size().reset_index()
            age_health_counts.columns = ['Tranche d\'âge', 'Condition de santé', 'Nombre de donneurs']
            
            age_health_fig = px.bar(
                age_health_counts, 
                x='Tranche d\'âge', 
                y='Nombre de donneurs',
                color='Condition de santé',
                barmode='group',
                title='Distribution des conditions de santé par tranche d\'âge'
            )
            st.plotly_chart(age_health_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la génération des visualisations: {str(e)}")

# Ajouter les autres fonctions d'affichage ici (display_donor_clustering, etc.)

def display_donor_clustering(df):
    st.header("Profilage des donateurs idéaux")
    st.markdown("Regroupement des donneurs en profils similaires en fonction de caractéristiques démographiques et de santé.")
    
    try:
        # Vérifier si les colonnes nécessaires existent
        required_columns = ['age', 'sexe', 'profession', 'condition_sante']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Les colonnes suivantes sont manquantes pour le clustering: {', '.join(missing_columns)}")
            st.info("Génération de visualisations alternatives...")
            
            # Créer des visualisations alternatives basées sur les colonnes disponibles
            if 'age' in df.columns and 'eligible' in df.columns:
                # Distribution par âge et éligibilité
                st.subheader("Distribution des donneurs par âge et éligibilité")
                age_elig_fig = px.histogram(
                    df, 
                    x='age',
                    color='eligible',
                    marginal='box',
                    labels={'age': 'Âge', 'eligible': 'Éligible'},
                    title='Distribution des âges par éligibilité',
                    color_discrete_map={0: 'red', 1: 'green'}
                )
                st.plotly_chart(age_elig_fig, use_container_width=True)
            
            if 'sexe' in df.columns and 'condition_sante' in df.columns:
                # Distribution par sexe et condition de santé
                st.subheader("Distribution des conditions de santé par sexe")
                gender_health_counts = df.groupby(['sexe', 'condition_sante']).size().reset_index()
                gender_health_counts.columns = ['Sexe', 'Condition de santé', 'Nombre de donneurs']
                
                gender_health_fig = px.bar(
                    gender_health_counts, 
                    x='Sexe', 
                    y='Nombre de donneurs',
                    color='Condition de santé',
                    barmode='group',
                    title='Distribution des conditions de santé par sexe'
                )
                st.plotly_chart(gender_health_fig, use_container_width=True)
            
            return
        
        with st.spinner("Calcul des clusters..."):
            fig, cluster_profiles = donor_clustering(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les profils de clusters
        st.subheader("Caractéristiques des groupes de donneurs identifiés")
        
        # Créer une mise en page en colonnes pour les profils
        cols = st.columns(len(cluster_profiles))
        
        for i, (col, profile) in enumerate(zip(cols, cluster_profiles)):
            with col:
                st.write(f"**Groupe {i+1}**")
                st.write(f"**Nombre de donneurs:** {profile['count']}")
                st.write(f"**Âge moyen:** {profile['age_mean']:.1f} ans")
                st.write(f"**Profession dominante:** {profile['top_profession']}")
                st.write(f"**Condition de santé dominante:** {profile['top_health_condition']}")
                st.write("**Répartition par sexe:**")
                for gender, ratio in profile['gender_ratio'].items():
                    st.write(f"- {gender}: {ratio:.1%}")
        
        # Ajouter des visualisations supplémentaires pour comparer les clusters
        st.subheader("Comparaison des clusters")
        
        # Extraire les données pour la visualisation
        cluster_data = []
        for i, profile in enumerate(cluster_profiles):
            cluster_data.append({
                'Cluster': f"Groupe {i+1}",
                'Nombre de donneurs': profile['count'],
                'Âge moyen': profile['age_mean']
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Graphique à barres pour comparer les clusters
        cluster_compare_fig = px.bar(
            cluster_df,
            x='Cluster',
            y='Nombre de donneurs',
            color='Âge moyen',
            color_continuous_scale='Reds',
            title='Comparaison des clusters par taille et âge moyen'
        )
        st.plotly_chart(cluster_compare_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors du clustering: {str(e)}")
        st.info("Le clustering nécessite des données numériques et catégorielles valides.")

def display_campaign_effectiveness(df):
    st.header("Efficacité des campagnes")
    st.markdown("Analyse des tendances temporelles et des facteurs influençant le succès des campagnes de don de sang.")
    
    try:
        # Vérifier si la colonne de date existe
        if 'date_don' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date_don']):
            st.warning("La colonne 'date_don' n'existe pas ou n'est pas au format date dans les données.")
            st.info("Cette analyse nécessite des données temporelles pour être pertinente.")
            
            # Afficher d'autres visualisations pertinentes si possible
            if 'profession' in df.columns and 'eligible' in df.columns:
                # Distribution par profession et éligibilité
                st.subheader("Taux d'éligibilité par profession")
                prof_elig = df.groupby('profession').agg({
                    'id_donneur': 'count',
                    'eligible': 'mean'
                }).reset_index()
                prof_elig.columns = ['Profession', 'Nombre de donneurs', 'Taux d\'éligibilité']
                prof_elig['Taux d\'éligibilité'] = (prof_elig['Taux d\'éligibilité'] * 100).round(1)
                prof_elig = prof_elig.sort_values('Nombre de donneurs', ascending=False).head(10)
                
                prof_elig_fig = px.bar(
                    prof_elig,
                    x='Profession',
                    y='Nombre de donneurs',
                    color='Taux d\'éligibilité',
                    color_continuous_scale='Reds',
                    title='Top 10 des professions par nombre de donneurs et taux d\'éligibilité'
                )
                st.plotly_chart(prof_elig_fig, use_container_width=True)
            
            return
            
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
        
        # Ajouter une analyse supplémentaire: taux d'éligibilité par période
        if 'mois_don' in df.columns and 'eligible' in df.columns:
            st.subheader("Taux d'éligibilité par mois")
            month_elig = df.groupby('mois_don').agg({
                'id_donneur': 'count',
                'eligible': 'mean'
            }).reset_index()
            month_elig.columns = ['Mois', 'Nombre de donneurs', 'Taux d\'éligibilité']
            month_elig['Taux d\'éligibilité'] = (month_elig['Taux d\'éligibilité'] * 100).round(1)
            
            # Ajouter les noms des mois
            mois_noms = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin',
                        7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
            month_elig['Nom du mois'] = month_elig['Mois'].map(mois_noms)
            month_elig = month_elig.sort_values('Mois')
            
            month_elig_fig = px.line(
                month_elig,
                x='Nom du mois',
                y='Taux d\'éligibilité',
                markers=True,
                title='Taux d\'éligibilité par mois'
            )
            st.plotly_chart(month_elig_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des campagnes: {str(e)}")

def display_donor_retention(df):
    st.header("Fidélisation des donateurs")
    st.markdown("Analyse de la fidélité des donneurs et des facteurs démographiques associés aux dons répétés.")
    
    try:
        # Cette analyse est difficile à réaliser sans identifiants uniques répétés
        # Nous allons simuler des données de fidélisation si nécessaire
        if 'id_donneur' not in df.columns or df['id_donneur'].nunique() == len(df):
            st.warning("Les données ne contiennent pas d'information sur les dons répétés.")
            st.info("Pour cette démonstration, nous allons simuler des données de fidélisation.")
            
            # Simuler des données de fidélisation
            if 'age' in df.columns and 'sexe' in df.columns and 'profession' in df.columns:
                # Créer un DataFrame fictif de fidélisation
                retention_data = []
                for age_group in ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']:
                    for gender in df['sexe'].unique():
                        # Simuler un taux de fidélisation différent par âge et sexe
                        if age_group in ['25-34', '35-44', '45-54']:
                            retention_rate = np.random.uniform(0.6, 0.8)
                        else:
                            retention_rate = np.random.uniform(0.3, 0.6)
                            
                        # Légère différence par sexe
                        if gender == 'Homme':
                            retention_rate *= 0.9
                            
                        retention_data.append({
                            'Tranche d\'âge': age_group,
                            'Sexe': gender,
                            'Taux de fidélisation': retention_rate
                        })
                
                retention_df = pd.DataFrame(retention_data)
                
                # Visualiser le taux de fidélisation par tranche d'âge et sexe
                st.subheader("Taux de fidélisation simulé par tranche d'âge et sexe")
                retention_fig = px.bar(
                    retention_df,
                    x='Tranche d\'âge',
                    y='Taux de fidélisation',
                    color='Sexe',
                    barmode='group',
                    title='Taux de fidélisation simulé par tranche d\'âge et sexe'
                )
                st.plotly_chart(retention_fig, use_container_width=True)
                
                # Simulation pour les professions
                top_professions = df['profession'].value_counts().head(8).index.tolist()
                prof_retention_data = []
                
                for profession in top_professions:
                    # Simuler un taux de fidélisation différent par profession
                    retention_rate = np.random.uniform(0.4, 0.8)
                    prof_retention_data.append({
                        'Profession': profession,
                        'Taux de fidélisation': retention_rate
                    })
                
                prof_retention_df = pd.DataFrame(prof_retention_data)
                
                # Visualiser le taux de fidélisation par profession
                st.subheader("Taux de fidélisation simulé par profession")
                prof_retention_fig = px.bar(
                    prof_retention_df,
                    x='Profession',
                    y='Taux de fidélisation',
                    color='Taux de fidélisation',
                    color_continuous_scale='Reds',
                    title='Taux de fidélisation simulé par profession'
                )
                st.plotly_chart(prof_retention_fig, use_container_width=True)
            
            return
        
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
        # Vérifier si la colonne de commentaires existe
        comment_col = None
        for col in ['commentaire', 'Si_autres_raison_préciser_', 'Si autres raison préciser']:
            if col in df.columns:
                comment_col = col
                break
                
        if comment_col is None:
            st.warning("Aucune colonne de commentaires trouvée dans les données.")
            st.info("Cette analyse nécessite des commentaires textuels des donneurs.")
            return
        
        # Renommer la colonne si nécessaire
        df_copy = df.copy()
        if comment_col != 'commentaire':
            df_copy['commentaire'] = df_copy[comment_col]
        
        fig_dist, wordcloud_figs = sentiment_analysis(df_copy)
        
        st.subheader("Distribution des sentiments")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Afficher les wordclouds
        st.subheader("Mots-clés par sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (sentiment, fig) in enumerate(wordcloud_figs.items()):
            with [col1, col2, col3][i % 3]:
                st.write(f"**{sentiment}**")
                st.pyplot(fig)
        
        # Afficher des exemples de commentaires pour chaque catégorie
        st.subheader("Exemples de commentaires par sentiment")
        
        sentiments = ['Positif', 'Neutre', 'Négatif']
        for sentiment in sentiments:
            with st.expander(f"Exemples de commentaires {sentiment.lower()}s"):
                sentiment_df = df_copy[df_copy['sentiment'] == sentiment]
                if not sentiment_df.empty:
                    sample_comments = sentiment_df['commentaire'].dropna().sample(min(5, len(sentiment_df))).tolist()
                    for i, comment in enumerate(sample_comments, 1):
                        st.write(f"{i}. {comment}")
                else:
                    st.write(f"Aucun commentaire {sentiment.lower()} trouvé.")
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des sentiments: {str(e)}")
        st.info("L'analyse des sentiments nécessite des données textuelles. Assurez-vous que la colonne 'commentaire' existe.")

def display_eligibility_prediction(df):
    st.header("Prédiction d'éligibilité au don de sang")
    st.markdown("""
    Utilisez ce formulaire pour prédire si une personne est éligible au don de sang
    en fonction de ses caractéristiques personnelles et de santé.
    """)
    
    # Vérifier si les colonnes nécessaires existent
    required_columns = ['age', 'sexe', 'profession', 'condition_sante', 'eligible']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Les colonnes suivantes sont manquantes pour l'entraînement du modèle: {', '.join(missing_columns)}")
        st.info("Cette fonctionnalité nécessite des données complètes pour l'entraînement du modèle.")
        return
    
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
                
                # Afficher des informations supplémentaires
                st.info("""
                **Informations supplémentaires:**
                - Le donneur est éligible au don de sang selon notre modèle.
                - Le don de sang est un acte citoyen qui peut sauver jusqu'à 3 vies.
                - Un donneur peut donner son sang toutes les 8 semaines (soit 6 fois par an maximum).
                """)
            else:
                st.error(f"❌ Non éligible au don de sang (Probabilité: {result['probability']:.2f})")
                
                # Afficher des facteurs possibles d'inéligibilité
                st.warning("""
                **Facteurs possibles d'inéligibilité:**
                - Le donneur présente une condition de santé qui peut contre-indiquer le don.
                - Certaines conditions nécessitent un délai d'attente avant de pouvoir donner son sang.
                - La sécurité du donneur et du receveur est la priorité absolue.
                """)
            
            st.info("Note: Cette prédiction est basée sur un modèle d'apprentissage automatique et ne remplace pas l'avis d'un professionnel de santé.")
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    main()