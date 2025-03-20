import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
import os
import streamlit as st

# Télécharger les ressources nltk si nécessaire
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def create_donor_map(df):
    """
    Crée une carte interactive de la répartition des donneurs.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        folium.Map: Carte Folium interactive
    """
    # Créer une carte centrée (utiliser des coordonnées fictives pour la démonstration)
    # Dans un cas réel, vous utiliseriez les coordonnées réelles des arrondissements
    donor_map = folium.Map(location=[0, 0], zoom_start=12)
    
    # Pour la démonstration, créons des coordonnées fictives pour chaque arrondissement
    arrondissements = df['arrondissement'].unique()
    n_arrond = len(arrondissements)
    
    # Créer une grille de coordonnées pour les arrondissements
    coords = []
    for i in range(n_arrond):
        row = i // 3
        col = i % 3
        coords.append((row * 0.02, col * 0.02))
    
    arrond_coords = dict(zip(arrondissements, coords))
    
    # Calculer les statistiques par arrondissement
    arrond_stats = df.groupby('arrondissement').agg({
        'id_donneur': 'count',
        'eligible': 'mean'
    }).reset_index()
    
    # Créer des marqueurs pour chaque arrondissement
    for _, row in arrond_stats.iterrows():
        arrond = row['arrondissement']
        count = row['id_donneur']
        eligible_rate = row['eligible']
        
        if arrond in arrond_coords:
            folium.CircleMarker(
                location=arrond_coords[arrond],
                radius=count / 10 if count < 100 else 10,  # Limiter la taille
                popup=f"<strong>{arrond}</strong><br>Donneurs: {count}<br>Taux d'éligibilité: {eligible_rate:.1%}",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6
            ).add_to(donor_map)
    
    return donor_map

def health_conditions_chart(df):
    """
    Crée un graphique à barres des conditions de santé des donneurs.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        plotly.graph_objects.Figure: Graphique Plotly
    """
    # Compter les occurrences de chaque condition de santé
    condition_counts = df['condition_sante'].value_counts().reset_index()
    condition_counts.columns = ['condition', 'count']
    
    # Trier par nombre décroissant
    condition_counts = condition_counts.sort_values('count', ascending=False)
    
    # Créer un graphique à barres
    fig = px.bar(
        condition_counts, 
        x='condition', 
        y='count',
        title='Répartition des conditions de santé parmi les donneurs',
        labels={'condition': 'Condition de santé', 'count': 'Nombre de donneurs'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    # Personnaliser la mise en page
    fig.update_layout(
        xaxis_title='Condition de santé',
        yaxis_title='Nombre de donneurs',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def eligibility_by_condition(df):
    """
    Crée un graphique montrant l'éligibilité par condition de santé.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        plotly.graph_objects.Figure: Graphique Plotly
    """
    # Grouper par condition de santé et éligibilité
    condition_elig = df.groupby(['condition_sante', 'eligible']).size().reset_index()
    condition_elig.columns = ['condition', 'eligible', 'nombre']
    
    # Pivoter pour obtenir les admissibles et non admissibles côte à côte
    pivot_df = condition_elig.pivot(index='condition', columns='eligible', values='nombre').fillna(0)
    
    # Renommer les colonnes
    if 0 in pivot_df.columns and 1 in pivot_df.columns:
        pivot_df.columns = ['Non admissible', 'Admissible']
    else:
        # Gérer le cas où toutes les personnes sont éligibles ou non éligibles
        if 0 in pivot_df.columns:
            pivot_df['Admissible'] = 0
            pivot_df = pivot_df[['Non admissible', 'Admissible']]
        else:
            pivot_df['Non admissible'] = 0
            pivot_df = pivot_df[['Non admissible', 'Admissible']]
    
    pivot_df.reset_index(inplace=True)
    
    # Calculer le pourcentage d'admissibilité
    pivot_df['Total'] = pivot_df['Admissible'] + pivot_df['Non admissible']
    pivot_df['Pourcentage admissible'] = (pivot_df['Admissible'] / pivot_df['Total'] * 100).round(1)
    
    # Créer un graphique à barres groupées
    fig = px.bar(
        pivot_df, 
        x='condition', 
        y=['Admissible', 'Non admissible'],
        barmode='group', 
        title='Admissibilité par condition de santé',
        labels={'condition': 'Condition de santé', 'value': 'Nombre de donneurs', 'variable': 'Statut'},
        color_discrete_map={'Admissible': 'green', 'Non admissible': 'red'}
    )
    
    # Ajouter des annotations pour les pourcentages
    for i, row in enumerate(pivot_df.itertuples()):
        fig.add_annotation(
            x=row.condition,
            y=row.Total + 5,  # Légèrement au-dessus de la barre
            text=f"{row.Pourcentage_admissible}%",
            showarrow=False
        )
    
    # Personnaliser la mise en page
    fig.update_layout(
        xaxis_title='Condition de santé',
        yaxis_title='Nombre de donneurs',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def donor_clustering(df):
    """
    Effectue un clustering des donneurs et visualise les résultats.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        tuple: (figure Plotly, liste des profils de clusters)
    """
    # Sélectionner les caractéristiques pour le clustering
    features = ['age', 'sexe', 'profession', 'condition_sante']
    
    # Vérifier que toutes les colonnes nécessaires existent
    for col in features:
        if col not in df.columns:
            raise ValueError(f"La colonne {col} est nécessaire pour le clustering mais n'existe pas dans les données")
    
    # Créer des copies pour éviter les avertissements de modification
    X = df[features].copy()
    
    # Préparer les transformateurs pour les variables numériques et catégorielles
    numeric_features = ['age']
    categorical_features = ['sexe', 'profession', 'condition_sante']
    
    # Vérifier le nombre de clusters optimal (entre 3 et 5)
    n_clusters = min(5, len(df) // 20) if len(df) > 60 else 3
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Créer un pipeline avec prétraitement et clustering
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    
    # Ajuster le modèle et obtenir les labels de cluster
    df_copy = df.copy()
    df_copy['cluster'] = pipeline.fit_predict(X)
    
    # Réduire la dimensionnalité pour la visualisation
    X_processed = preprocessor.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed.toarray())
    
    # Créer un DataFrame pour la visualisation
    plot_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': df_copy['cluster'].astype(str)
    })
    
    # Créer un graphique de dispersion
    fig = px.scatter(
        plot_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        title='Clusters de donneurs identifiés',
        labels={'Cluster': 'Groupe de donneur'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Extraire les caractéristiques de chaque cluster
    cluster_profiles = []
    for cluster_id in range(n_clusters):
        cluster_data = df_copy[df_copy['cluster'] == cluster_id]
        
        # Éviter les erreurs si le cluster est vide
        if len(cluster_data) == 0:
            continue
            
        profile = {
            'cluster_id': cluster_id,
            'count': len(cluster_data),
            'age_mean': cluster_data['age'].mean(),
            'top_profession': cluster_data['profession'].mode()[0] if not cluster_data['profession'].mode().empty else "N/A",
            'top_health_condition': cluster_data['condition_sante'].mode()[0] if not cluster_data['condition_sante'].mode().empty else "N/A",
            'gender_ratio': cluster_data['sexe'].value_counts(normalize=True).to_dict()
        }
        cluster_profiles.append(profile)
    
    return fig, cluster_profiles

def campaign_effectiveness(df):
    """
    Analyse et visualise l'efficacité des campagnes de don de sang.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        tuple: (figure de tendance temporelle, figure mensuelle, figure démographique)
    """
    # Vérifier que la colonne de date existe et est au format datetime
    if 'date_don' not in df.columns:
        raise ValueError("La colonne 'date_don' est nécessaire pour l'analyse temporelle")
    
    # Convertir la date en format datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['date_don']):
        df['date_don'] = pd.to_datetime(df['date_don'], errors='coerce')
    
    # Extraire le mois et l'année
    df['mois'] = df['date_don'].dt.month
    df['annee'] = df['date_don'].dt.year
    
    # Agréger par mois et année
    monthly_donations = df.groupby(['annee', 'mois']).size().reset_index()
    monthly_donations.columns = ['Année', 'Mois', 'Nombre de dons']
    
    # Créer une colonne de date pour le graphique
    monthly_donations['Date'] = pd.to_datetime(monthly_donations[['Année', 'Mois']].assign(day=1))
    
    # Créer un graphique linéaire
    fig_time = px.line(
        monthly_donations, 
        x='Date', 
        y='Nombre de dons',
        title='Évolution des dons de sang au fil du temps',
        labels={'Date': 'Date', 'Nombre de dons': 'Nombre de dons'},
        markers=True
    )
    
    # Agréger par mois (toutes années confondues)
    monthly_pattern = df.groupby('mois').size().reset_index()
    monthly_pattern.columns = ['Mois', 'Nombre de dons']
    
    # Ajouter les noms des mois
    mois_noms = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin',
                 7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
    monthly_pattern['Nom du mois'] = monthly_pattern['Mois'].map(mois_noms)
    
    # Trier par mois
    monthly_pattern = monthly_pattern.sort_values('Mois')
    
    # Créer un graphique à barres
    fig_month = px.bar(
        monthly_pattern, 
        x='Nom du mois', 
        y='Nombre de dons',
        title='Saisonnalité des dons de sang',
        labels={'Nom du mois': 'Mois', 'Nombre de dons': 'Nombre de dons'},
        color='Nombre de dons', 
        color_continuous_scale='Reds'
    )
    
    # Analyser par groupe démographique (profession)
    if 'profession' in df.columns:
        demo_effectiveness = df.groupby(['profession']).agg({
            'id_donneur': 'count'  # Compter le nombre de donneurs
        }).reset_index()
        demo_effectiveness.columns = ['Profession', 'Nombre de dons']
        demo_effectiveness = demo_effectiveness.sort_values('Nombre de dons', ascending=False)
        
        # Limiter aux 10 premières professions
        demo_effectiveness = demo_effectiveness.head(10)
        
        # Créer un graphique à barres
        fig_demo = px.bar(
            demo_effectiveness, 
            x='Profession', 
            y='Nombre de dons',
            title='Top 10 des professions des donneurs',
            labels={'Profession': 'Profession', 'Nombre de dons': 'Nombre de dons'},
            color='Nombre de dons', 
            color_continuous_scale='Reds'
        )
    else:
        # Créer un graphique vide si la colonne profession n'existe pas
        fig_demo = go.Figure()
        fig_demo.update_layout(
            title='Données de profession non disponibles',
            xaxis_title='Profession',
            yaxis_title='Nombre de dons'
        )
    
    return fig_time, fig_month, fig_demo

def donor_retention_analysis(df):
    """
    Analyse la fidélisation des donneurs.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        tuple: (figure de distribution de fidélité, figure par âge, figure par profession)
    """
    # Pour cette analyse, nous avons besoin d'un identifiant unique par donneur
    if 'id_donneur' not in df.columns:
        raise ValueError("La colonne 'id_donneur' est nécessaire pour l'analyse de fidélisation")
    
    # Calculer le nombre de dons par donateur (simulé pour la démonstration)
    # Dans un cas réel, on compterait les occurrences réelles de chaque donneur
    donor_counts = df['id_donneur'].value_counts().reset_index()
    donor_counts.columns = ['id_donneur', 'nombre_dons']
    
    # Fusionner avec les données démographiques
    donor_demo = df.drop_duplicates('id_donneur')
    if 'age' in donor_demo.columns and 'sexe' in donor_demo.columns and 'profession' in donor_demo.columns:
        donor_demo = donor_demo[['id_donneur', 'age', 'sexe', 'profession']]
    else:
        # Si les colonnes n'existent pas, créer un dataframe minimal
        donor_demo = donor_demo[['id_donneur']]
        if 'age' not in donor_demo.columns:
            donor_demo['age'] = np.random.randint(18, 70, size=len(donor_demo))
        if 'sexe' not in donor_demo.columns:
            donor_demo['sexe'] = np.random.choice(['Homme', 'Femme'], size=len(donor_demo))
        if 'profession' not in donor_demo.columns:
            professions = ['Étudiant', 'Enseignant', 'Médecin', 'Ingénieur', 'Commerçant', 'Retraité']
            donor_demo['profession'] = np.random.choice(professions, size=len(donor_demo))
            
    donor_retention = donor_demo.merge(donor_counts, on='id_donneur')
    
    # Créer des catégories de fidélité
    bins = [0, 1, 3, 5, float('inf')]
    labels = ['Unique', '2-3 dons', '4-5 dons', '6+ dons']
    donor_retention['categorie_fidelite'] = pd.cut(donor_retention['nombre_dons'], bins=bins, labels=labels, right=False)
    
    # Distribution des catégories de fidélité
    loyalty_dist = donor_retention['categorie_fidelite'].value_counts().reset_index()
    loyalty_dist.columns = ['Catégorie de fidélité', 'Nombre de donneurs']
    
    fig_loyalty = px.pie(
        loyalty_dist, 
        values='Nombre de donneurs', 
        names='Catégorie de fidélité',
        title='Distribution des donneurs par fidélité',
        color_discrete_sequence=px.colors.sequential.Reds
    )
    
    # Analyser la fidélité par groupe d'âge
    age_loyalty = donor_retention.groupby('age')['nombre_dons'].mean().reset_index()
    age_loyalty.columns = ['Âge', 'Nombre moyen de dons']
    
    fig_age = px.line(
        age_loyalty, 
        x='Âge', 
        y='Nombre moyen de dons',
        title='Fidélité selon l\'âge',
        labels={'Âge': 'Âge', 'Nombre moyen de dons': 'Nombre moyen de dons'},
        markers=True
    )
    
    # Fidélité par profession
    prof_loyalty = donor_retention.groupby('profession')['nombre_dons'].mean().reset_index()
    prof_loyalty.columns = ['Profession', 'Nombre moyen de dons']
    prof_loyalty = prof_loyalty.sort_values('Nombre moyen de dons', ascending=False)
    
    fig_prof = px.bar(
        prof_loyalty.head(10), 
        x='Profession', 
        y='Nombre moyen de dons',
        title='Top 10 des professions par fidélité',
        labels={'Profession': 'Profession', 'Nombre moyen de dons': 'Nombre moyen de dons'},
        color='Nombre moyen de dons', 
        color_continuous_scale='Reds'
    )
    
    return fig_loyalty, fig_age, fig_prof

def sentiment_analysis(df):
    """
    Effectue une analyse des sentiments sur les commentaires des donneurs.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des donneurs
        
    Returns:
        tuple: (figure de distribution des sentiments, dictionnaire de figures wordcloud)
    """
    # Vérifier si la colonne de commentaires existe
    if 'commentaire' not in df.columns:
        # Créer une colonne de commentaires fictifs pour la démonstration
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
            "Bon accueil mais trop de questions sur ma vie privée."
        ]
        df['commentaire'] = np.random.choice(comments, size=len(df))
    
    # Fonction pour déterminer le sentiment d'un texte
    def get_sentiment(text):
        if pd.isna(text) or text == '':
            return 'Neutre'
        
        # Analyser le sentiment avec TextBlob
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positif'
        elif polarity < -0.1:
            return 'Négatif'
        else:
            return 'Neutre'
    
    # Appliquer l'analyse de sentiment aux commentaires
    df_sentiment = df.copy()
    df_sentiment['sentiment'] = df_sentiment['commentaire'].apply(get_sentiment)
    
    # Distribution des sentiments
    sentiment_counts = df_sentiment['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Nombre']
    
    # Graphique de distribution des sentiments
    fig_dist = px.pie(
        sentiment_counts, 
        values='Nombre', 
        names='Sentiment',
        title='Distribution des sentiments dans les commentaires',
        color_discrete_map={'Positif': 'green', 'Neutre': 'gray', 'Négatif': 'red'}
    )
    
    # Créer un wordcloud pour chaque sentiment
    try:
        stopwords_list = set(stopwords.words('french'))
    except:
        stopwords_list = set()
    
    wordclouds = {}
    for sentiment in ['Positif', 'Neutre', 'Négatif']:
        texts = df_sentiment[df_sentiment['sentiment'] == sentiment]['commentaire']
        if not texts.empty:
            all_text = ' '.join([str(text) for text in texts if isinstance(text, str)])
            if all_text.strip():  # S'assurer qu'il y a du texte
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    stopwords=stopwords_list
                ).generate(all_text)
                
                # Créer une figure matplotlib
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Mots-clés des commentaires {sentiment.lower()}s')
                
                wordclouds[sentiment] = fig
    
    return fig_dist, wordclouds
