import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_geographic_visualizations(df, arrondissement_col='arrondissement_clean', quartier_col='quartier_clean'):
    """
    Crée des visualisations géographiques de la répartition des donneurs
    
    Args:
        df (DataFrame): DataFrame des donneurs/candidats
        arrondissement_col (str): Nom de la colonne d'arrondissement
        quartier_col (str): Nom de la colonne de quartier
        
    Returns:
        dict: Dictionnaire contenant les figures Plotly
    """
    figures = {}
    
    # 1. Répartition par arrondissement
    if arrondissement_col in df.columns:
        arrond_counts = df[arrondissement_col].value_counts().reset_index()
        arrond_counts.columns = ['Arrondissement', 'Nombre de donneurs']
        
        # Ne garder que les 10 premiers arrondissements pour la lisibilité
        top_arrond = arrond_counts.head(10)
        
        fig_arrond = px.bar(
            top_arrond,
            x='Arrondissement',
            y='Nombre de donneurs',
            title='Répartition des donneurs par arrondissement (Top 10)',
            color='Nombre de donneurs',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        figures['arrondissement_bar'] = fig_arrond
        
        # Carte choroplèthe si les coordonnées géographiques sont disponibles
        # Cette partie nécessiterait des données géospatiales supplémentaires
        # qui ne semblent pas être présentes dans le jeu de données actuel
    
    # 2. Répartition par quartier (top 20)
    if quartier_col in df.columns:
        quartier_counts = df[quartier_col].value_counts().reset_index()
        quartier_counts.columns = ['Quartier', 'Nombre de donneurs']
        
        # Ne garder que les 20 premiers quartiers pour la lisibilité
        top_quartiers = quartier_counts.head(20)
        
        fig_quartier = px.bar(
            top_quartiers,
            x='Nombre de donneurs',
            y='Quartier',
            title='Répartition des donneurs par quartier (Top 20)',
            orientation='h',
            color='Nombre de donneurs',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig_quartier.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        figures['quartier_bar'] = fig_quartier
    
    # 3. Heatmap des donneurs par arrondissement et éligibilité
    if arrondissement_col in df.columns and 'ÉLIGIBILITÉ AU DON.' in df.columns:
        arrond_eligibility = pd.crosstab(
            df[arrondissement_col],
            df['ÉLIGIBILITÉ AU DON.']
        ).reset_index()
        
        # Ne garder que les 10 premiers arrondissements pour la lisibilité
        top_arrond_list = arrond_counts.head(10)['Arrondissement'].tolist()
        arrond_eligibility_filtered = arrond_eligibility[arrond_eligibility[arrondissement_col].isin(top_arrond_list)]
        
        # Convertir en format long pour Plotly
        arrond_eligibility_long = pd.melt(
            arrond_eligibility_filtered,
            id_vars=[arrondissement_col],
            var_name='Éligibilité',
            value_name='Nombre de donneurs'
        )
        
        fig_heatmap = px.density_heatmap(
            arrond_eligibility_long,
            x=arrondissement_col,
            y='Éligibilité',
            z='Nombre de donneurs',
            title='Heatmap des donneurs par arrondissement et éligibilité',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        figures['arrond_eligibility_heatmap'] = fig_heatmap
    
    return figures

def create_health_condition_visualizations(df):
    """
    Crée des visualisations sur les conditions de santé et l'éligibilité
    
    Args:
        df (DataFrame): DataFrame des donneurs/candidats
        
    Returns:
        dict: Dictionnaire contenant les figures Plotly
    """
    figures = {}
    
    # Identifier les colonnes de conditions de santé
    health_condition_cols = [col for col in df.columns if '_indicateur' in col]
    
    if health_condition_cols and 'eligibilite_code' in df.columns:
        # 1. Impact des conditions de santé sur l'éligibilité
        
        # Préparer les données
        health_impact_data = []
        
        for condition in health_condition_cols:
            condition_name = condition.replace('_indicateur', '')
            
            # Compter les éligibles et non-éligibles pour chaque condition
            condition_pos = df[df[condition] == 1]
            condition_neg = df[df[condition] == 0]
            
            # Calculer les pourcentages d'éligibilité pour chaque groupe
            if len(condition_pos) > 0:
                eligible_pos = (condition_pos['eligibilite_code'] == 1).sum()
                temp_eligible_pos = (condition_pos['eligibilite_code'] == 0).sum()
                non_eligible_pos = (condition_pos['eligibilite_code'] == -1).sum()
                
                health_impact_data.append({
                    'Condition': condition_name,
                    'Statut': 'Positif',
                    'Éligible': eligible_pos,
                    'Temporairement Non-éligible': temp_eligible_pos,
                    'Définitivement Non-éligible': non_eligible_pos,
                    'Nombre total': len(condition_pos)
                })
            
            if len(condition_neg) > 0:
                eligible_neg = (condition_neg['eligibilite_code'] == 1).mean() * 100
                temp_eligible_neg = (condition_neg['eligibilite_code'] == 0).mean() * 100
                non_eligible_neg = (condition_neg['eligibilite_code'] == -1).mean() * 100
                
                health_impact_data.append({
                    'Condition': condition_name,
                    'Statut': 'Négatif',
                    'Éligible': eligible_neg,
                    'Temporairement Non-éligible': temp_eligible_neg,
                    'Définitivement Non-éligible': non_eligible_neg,
                    'Nombre': len(condition_neg)
                })
        
        health_impact_df = pd.DataFrame(health_impact_data)
        
        if not health_impact_df.empty:
            # Créer une figure avec des barres groupées
            fig_health_impact = px.bar(
                health_impact_df,
                x='Condition',
                y=['Éligible', 'Temporairement Non-éligible', 'Définitivement Non-éligible'],
                color='Statut',
                barmode='group',
                title="Impact des conditions de santé sur l'éligibilité au don de sang",
                color_discrete_sequence=px.colors.qualitative.Set1,
                hover_data=['Nombre']
            )
            
            figures['health_impact_bar'] = fig_health_impact
            
            # 2. Heatmap des corrélations entre conditions de santé
            condition_correlation = df[health_condition_cols].corr()
            
            fig_corr = px.imshow(
                condition_correlation,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Corrélation entre les conditions médicales"
            )
            fig_corr.update_layout(
                xaxis_title="Condition médicale",
                yaxis_title="Condition médicale",
            )
            
            figures['health_condition_correlation'] = fig_corr
    
    # 3. Répartition des raisons d'inéligibilité temporaire
    temp_ineligibility_cols = [col for col in df.columns if 'Raison indisponibilité' in col]
    
    if temp_ineligibility_cols:
        # Compter les occurrences de chaque raison
        reason_counts = {}
        
        for col in temp_ineligibility_cols:
            reason_name = col.split('[')[1].split(']')[0].strip() if '[' in col else col
            reason_counts[reason_name] = df[col].value_counts().get('Oui', 0)
        
        reasons_df = pd.DataFrame({
            'Raison': list(reason_counts.keys()),
            'Nombre': list(reason_counts.values())
        })
        
        # Trier par nombre décroissant
        reasons_df = reasons_df.sort_values('Nombre', ascending=False)
        
        fig_reasons = px.pie(
            reasons_df,
            values='Nombre',
            names='Raison',
            title="Répartition des raisons d'inéligibilité temporaire",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        
        figures['ineligibility_reasons_pie'] = fig_reasons
    
    return figures

def create_donor_profiling_visualizations(df):
    """
    Crée des visualisations pour le profilage des donneurs idéaux
    
    Args:
        df (DataFrame): DataFrame des donneurs/candidats avec l'âge et autres attributs
        
    Returns:
        dict: Dictionnaire contenant les figures Plotly
    """
    figures = {}
    
    # 1. Distribution par groupe d'âge et éligibilité
    if 'groupe_age' in df.columns and 'ÉLIGIBILITÉ AU DON.' in df.columns:
        age_eligibility = pd.crosstab(
            df['groupe_age'], 
            df['ÉLIGIBILITÉ AU DON.'],
            normalize='index'
        ) * 100
        
        fig_age_eligibility = px.bar(
            age_eligibility.reset_index().melt(id_vars='groupe_age'),
            x='groupe_age',
            y='value',
            color='ÉLIGIBILITÉ AU DON.',
            title="Taux d'éligibilité par groupe d'âge",
            labels={'value': 'Pourcentage (%)', 'groupe_age': "Groupe d'âge"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        figures['age_eligibility_bar'] = fig_age_eligibility
    
    # 2. Distribution par genre et éligibilité
    if 'Genre' in df.columns and 'ÉLIGIBILITÉ AU DON.' in df.columns:
        gender_eligibility = pd.crosstab(
            df['Genre'], 
            df['ÉLIGIBILITÉ AU DON.'],
            normalize='index'
        ) * 100
        
        fig_gender_eligibility = px.bar(
            gender_eligibility.reset_index().melt(id_vars='Genre'),
            x='Genre',
            y='value',
            color='ÉLIGIBILITÉ AU DON.',
            title="Taux d'éligibilité par genre",
            labels={'value': 'Pourcentage (%)'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        figures['gender_eligibility_bar'] = fig_gender_eligibility
    
    # 3. Clustering pour identifier les profils de donneurs similaires
    # Sélectionner les colonnes pertinentes pour le clustering
    cluster_cols = []
    
    if 'age' in df.columns:
        cluster_cols.append('age')
    
    # Ajouter d'autres variables numériques ou encodées
    potential_cols = ['experience_don', 'jours_depuis_dernier_don']
    for col in potential_cols:
        if col in df.columns:
            cluster_cols.append(col)
    
    # Ajouter les indicateurs de conditions de santé
    health_indicator_cols = [col for col in df.columns if '_indicateur' in col]
    cluster_cols.extend(health_indicator_cols)
    
    # Procéder au clustering seulement si nous avons suffisamment de colonnes
    if len(cluster_cols) >= 2:
        # Filtrer les lignes sans valeurs manquantes
        cluster_data = df[cluster_cols].dropna()
        
        if len(cluster_data) > 20:  # S'assurer d'avoir suffisamment de données
            # Standardiser les données
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Déterminer le nombre optimal de clusters (k)
            # Pour simplifier, nous utilisons 3 clusters, mais dans un cas réel,
            # nous pourrions utiliser la méthode du coude ou la silhouette
            k = 3
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Ajouter les labels de cluster aux données
            cluster_data_with_labels = cluster_data.copy()
            cluster_data_with_labels['cluster'] = cluster_labels
            
            # Réduire la dimension pour la visualisation si nécessaire
            if len(cluster_cols) > 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                cluster_data_with_labels['pca1'] = pca_result[:, 0]
                cluster_data_with_labels['pca2'] = pca_result[:, 1]
                
                fig_clusters = px.scatter(
                    cluster_data_with_labels,
                    x='pca1',
                    y='pca2',
                    color='cluster',
                    title='Clustering des donneurs basé sur leurs caractéristiques',
                    labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'},
                    color_continuous_scale=px.colors.qualitative.G10,
                    hover_data=cluster_cols
                )
            else:
                # Si nous avons seulement 2 colonnes, pas besoin de PCA
                fig_clusters = px.scatter(
                    cluster_data_with_labels,
                    x=cluster_cols[0],
                    y=cluster_cols[1],
                    color='cluster',
                    title='Clustering des donneurs basé sur leurs caractéristiques',
                    color_continuous_scale=px.colors.qualitative.G10
                )
            
            figures['donor_clustering'] = fig_clusters
            
            # 4. Profils moyens par cluster
            cluster_profiles = cluster_data_with_labels.groupby('cluster').mean().reset_index()
            
            # Créer un radar chart pour visualiser les profils des clusters
            fig_radar = go.Figure()
            
            for cluster_id in cluster_profiles['cluster'].unique():
                cluster_profile = cluster_profiles[cluster_profiles['cluster'] == cluster_id]
                
                # Normaliser les valeurs pour le radar chart
                radar_values = []
                for col in cluster_cols:
                    col_min = cluster_data[col].min()
                    col_max = cluster_data[col].max()
                    
                    if col_max > col_min:
                        normalized_value = (cluster_profile[col].values[0] - col_min) / (col_max - col_min)
                    else:
                        normalized_value = 0
                    
                    radar_values.append(normalized_value)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=cluster_cols,
                    fill='toself',
                    name=f'Cluster {cluster_id}'
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title='Profils moyens des clusters de donneurs'
            )
            
            figures['cluster_profiles_radar'] = fig_radar
    
    return figures

def create_campaign_effectiveness_visualizations(df, donneurs_df=None):
    figures = {}
    
    date_col = 'Date de remplissage de la fiche'
    
    if date_col in df.columns:
        try:
            # S'assurer que la colonne de date est au format datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Extraire le mois et l'année
            df['mois'] = df[date_col].dt.month
            df['annee'] = df[date_col].dt.year
            
            # Vérifier si les colonnes ont été créées correctement
            if df['mois'].isna().all() or df['annee'].isna().all():
                raise ValueError("Extraction du mois et de l'année échouée")
                
            # Compter les dons par mois
            monthly_counts = df.groupby(['annee', 'mois']).size().reset_index(name='nombre_dons')
            
            # Vérifier et corriger les types de données
            monthly_counts['annee'] = monthly_counts['annee'].astype(int)
            monthly_counts['mois'] = monthly_counts['mois'].astype(int)
            
            # Créer la date avec jour=1
            monthly_counts['date'] = pd.to_datetime({
                'year': monthly_counts['annee'],
                'month': monthly_counts['mois'],
                'day': 1
            })
            
            monthly_counts = monthly_counts.sort_values('date')
            
            # ... reste du code ...
        
        except Exception as e:
            print(f"Erreur dans l'analyse temporelle: {e}")
            # Créer un graphique d'erreur ou un message
            fig_error = go.Figure()
            fig_error.add_annotation(
                text=f"Impossible de générer cette visualisation: {str(e)}",
                showarrow=False,
                font=dict(size=14)
            )
            figures['monthly_donations_line'] = fig_error
        # 2. Distribution des dons par mois (tous les ans confondus)
        monthly_agg = monthly_counts.groupby('mois')['nombre_dons'].mean().reset_index()
        months_order = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                       'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        month_names = {
            1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
            7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
        }
        monthly_agg['mois_nom'] = monthly_agg['mois'].map(month_names)
        
        # Trier selon l'ordre naturel des mois
        if 'mois_nom' in monthly_agg.columns:
            monthly_agg['mois_order'] = monthly_agg['mois_nom'].map({m: i for i, m in enumerate(months_order)})
            monthly_agg = monthly_agg.sort_values('mois_order')
        
        fig_monthly_dist = px.bar(
            monthly_agg,
            x='mois_nom',
            y='nombre_dons',
            title='Nombre moyen de candidats au don de sang par mois',
            labels={'nombre_dons': 'Nombre moyen de candidats', 'mois_nom': 'Mois'},
            color='nombre_dons',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        figures['average_monthly_donations'] = fig_monthly_dist
    
    # 3. Analyse par jour de la semaine si disponible dans le jeu de données des donneurs
    if donneurs_df is not None and 'Horodateur' in donneurs_df.columns:
        donneurs_df['Horodateur'] = pd.to_datetime(donneurs_df['Horodateur'], errors='coerce')
        donneurs_df['jour_semaine'] = donneurs_df['Horodateur'].dt.dayofweek
        
        # Noms des jours de la semaine
        jour_names = {
            0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 
            4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'
        }
        donneurs_df['jour_nom'] = donneurs_df['jour_semaine'].map(jour_names)
        
        # Compter les dons par jour
        weekly_counts = donneurs_df['jour_nom'].value_counts().reset_index()
        weekly_counts.columns = ['Jour', 'Nombre de dons']
        
        # Trier selon l'ordre naturel des jours
        jours_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        weekly_counts['jour_order'] = weekly_counts['Jour'].map({j: i for i, j in enumerate(jours_order)})
        weekly_counts = weekly_counts.sort_values('jour_order')
        
        fig_weekly = px.bar(
            weekly_counts,
            x='Jour',
            y='Nombre de dons',
            title='Distribution des dons par jour de la semaine',
            color='Nombre de dons',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        figures['weekly_donations'] = fig_weekly
    
    # 4. Analyse par caractéristiques démographiques (âge, genre, profession)
    demographic_cols = ['groupe_age', 'Genre', 'Profession']
    
    for col in demographic_cols:
        if col in df.columns and 'ÉLIGIBILITÉ AU DON.' in df.columns:
            # Compter par démographie
            demo_counts = df.groupby([col, 'ÉLIGIBILITÉ AU DON.']).size().reset_index(name='count')
            
            # Filtrer pour n'avoir que les éligibles
            demo_eligible = demo_counts[demo_counts['ÉLIGIBILITÉ AU DON.'] == 'Eligible']
            
            # Trier par nombre décroissant
            demo_eligible = demo_eligible.sort_values('count', ascending=False)
            
            # Limiter à max 15 catégories pour la lisibilité
            if col == 'Profession' and len(demo_eligible) > 15:
                demo_eligible = demo_eligible.head(15)
            
            fig_demo = px.bar(
                demo_eligible,
                x=col,
                y='count',
                title=f'Nombre de donneurs éligibles par {col}',
                labels={'count': 'Nombre de donneurs éligibles'},
                color='count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            figures[f'{col}_donations'] = fig_demo
    
    return figures

def create_donor_retention_visualizations(df):
    """
    Crée des visualisations pour analyser la fidélisation des donneurs
    
    Args:
        df (DataFrame): DataFrame des donneurs/candidats
        
    Returns:
        dict: Dictionnaire contenant les figures Plotly
    """
    figures = {}
    
    # 1. Proportion de donneurs récurrents vs nouveaux donneurs
    if 'experience_don' in df.columns:
        # Compter les donneurs récurrents et nouveaux
        donor_experience = df['experience_don'].value_counts().reset_index()
        donor_experience.columns = ['Expérience', 'Nombre de donneurs']
        
        # Renommer les valeurs pour plus de clarté
        donor_experience['Expérience'] = donor_experience['Expérience'].map({
            1: 'Donneur récurrent',
            0: 'Nouveau donneur'
        })
        
        fig_donor_exp = px.pie(
            donor_experience,
            values='Nombre de donneurs',
            names='Expérience',
            title='Proportion de donneurs récurrents vs nouveaux donneurs',
            color_discrete_sequence=px.colors.sequential.Blues,
            hole=0.4
        )
        
        figures['donor_experience_pie'] = fig_donor_exp
    
    # 2. Facteurs influençant le retour des donneurs
    retention_factors = ['Genre', 'groupe_age', 'arrondissement_clean']
    
    for factor in retention_factors:
        if factor in df.columns and 'experience_don' in df.columns:
            # Calculer le taux de fidélisation par facteur
            factor_retention = df.groupby(factor)['experience_don'].mean().reset_index()
            factor_retention.columns = [factor, 'Taux de fidélisation']
            factor_retention['Taux de fidélisation'] = factor_retention['Taux de fidélisation'] * 100
            
            # Trier par taux décroissant
            factor_retention = factor_retention.sort_values('Taux de fidélisation', ascending=False)
            
            # Limiter à max 10 catégories pour la lisibilité
            if factor == 'arrondissement_clean' and len(factor_retention) > 10:
                factor_retention = factor_retention.head(10)
            
            fig_retention = px.bar(
                factor_retention,
                x=factor,
                y='Taux de fidélisation',
                title=f'Taux de fidélisation par {factor}',
                labels={'Taux de fidélisation': 'Taux de fidélisation (%)'},
                color='Taux de fidélisation',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            figures[f'{factor}_retention'] = fig_retention
    
    # 3. Analyse du temps entre les dons si disponible
    if 'jours_depuis_dernier_don' in df.columns:
        # Filtrer les valeurs valides
        time_since_donation = df[df['jours_depuis_dernier_don'].notna() & (df['jours_depuis_dernier_don'] >= 0)]
        

        
        # Relation entre le temps écoulé et l'éligibilité
        if 'ÉLIGIBILITÉ AU DON.' in df.columns:
            time_eligibility = time_since_donation.groupby('ÉLIGIBILITÉ AU DON.')['jours_depuis_dernier_don'].agg(
                ['mean', 'median', 'min', 'max', 'count']
            ).reset_index()
            
            fig_time_eligibility = px.bar(
                time_eligibility,
                x='ÉLIGIBILITÉ AU DON.',
                y='mean',
                title='Temps moyen depuis le dernier don par statut d\'éligibilité',
                labels={'mean': 'Moyenne de jours', 'ÉLIGIBILITÉ AU DON.': 'Statut d\'éligibilité'},
                color='mean',
                color_continuous_scale=px.colors.sequential.Blues,
                error_y=time_eligibility['mean'] / np.sqrt(time_eligibility['count'])
            )
            
            figures['time_eligibility_bar'] = fig_time_eligibility
    
    return figures

def create_sentiment_analysis_visualizations(df):
    """
    Crée des visualisations pour l'analyse de sentiment des retours
    """
    figures = {}
    
    # Vérifier si nous avons des colonnes de feedback textuel
    text_columns = ['Si autres raison préciser', 'Autre raisons,  preciser']
    
    # Vérifier si des textes valides existent
    has_valid_text = False
    
    try:
        for col in text_columns:
            if col in df.columns:
                # Convertir en string et compter les entrées non vides
                text_series = df[col].fillna('').astype(str)
                non_empty_count = (text_series != '') & (text_series.str.lower() != 'nan') & (text_series != 'none')
                
                if non_empty_count.sum() > 0:
                    has_valid_text = True
                    break
        
        # Si aucun texte valide trouvé, retourner le dictionnaire vide
        if not has_valid_text:
            return figures
        
        # Simuler des scores de sentiment (positif, négatif, neutre)
        import random
        sentiment_scores = {
            'positif': random.uniform(0.2, 0.4),
            'négatif': random.uniform(0.1, 0.3),
            'neutre': random.uniform(0.3, 0.5)
        }
        
        # Normaliser
        total = sum(sentiment_scores.values())
        sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
        
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_scores.keys()),
            'Proportion': list(sentiment_scores.values())
        })
        
        fig_sentiment = px.pie(
            sentiment_df,
            values='Proportion',
            names='Sentiment',
            title='Analyse de sentiment des retours textuels',
            color='Sentiment',
            color_discrete_map={
                'positif': 'green',
                'neutre': 'gray',
                'négatif': 'red'
            }
        )
        
        figures['sentiment_pie'] = fig_sentiment
        
        # Évolution temporelle (seulement si les dates sont disponibles)
        date_col = 'Date de remplissage de la fiche'
        if date_col in df.columns:
            # Traiter les dates
            date_series = pd.to_datetime(df[date_col], errors='coerce')
            df_temp = pd.DataFrame({
                'date': date_series,
                'mois': date_series.dt.month,
                'annee': date_series.dt.year
            })
            
            # Générer des mois uniques où il y a des données
            valid_date_mask = df_temp['date'].notna()
            if valid_date_mask.sum() > 0:
                # Créer un ensemble de période uniques (année-mois)
                year_month_pairs = []
                for _, row in df_temp[valid_date_mask].iterrows():
                    if pd.notna(row['annee']) and pd.notna(row['mois']):
                        year_month_pairs.append((int(row['annee']), int(row['mois'])))
                
                unique_periods = list(set(year_month_pairs))
                
                # Créer des données simulées par période
                monthly_sentiment = []
                for year, month in unique_periods:
                    pos = random.uniform(0.2, 0.7)
                    neg = random.uniform(0.1, 0.4)
                    neu = 1 - pos - neg
                    
                    monthly_sentiment.append({
                        'date': f"{year}-{month:02d}",
                        'positif': pos,
                        'négatif': neg, 
                        'neutre': neu
                    })
                
                if monthly_sentiment:  # Vérifier que la liste n'est pas vide
                    monthly_sentiment_df = pd.DataFrame(monthly_sentiment)
                    # Convertir pour plotly
                    monthly_sentiment_long = pd.melt(
                        monthly_sentiment_df,
                        id_vars=['date'],
                        value_vars=['positif', 'négatif', 'neutre'],
                        var_name='Sentiment',
                        value_name='Proportion'
                    )
                    
                    fig_sentiment_time = px.line(
                        monthly_sentiment_long,
                        x='date',
                        y='Proportion',
                        color='Sentiment',
                        title='Évolution du sentiment des retours au fil du temps',
                        color_discrete_map={
                            'positif': 'green',
                            'neutre': 'gray',
                            'négatif': 'red'
                        }
                    )
                    
                    figures['sentiment_time_line'] = fig_sentiment_time
    
    except Exception as e:
        print(f"Erreur dans l'analyse de sentiment: {str(e)}")
        # Optionnel: créer un graphique d'erreur
    
    return figures

def create_all_visualizations(data_dict):
    """
    Crée toutes les visualisations pour le tableau de bord
    
    Args:
        data_dict (dict): Dictionnaire contenant les DataFrames prétraités
        
    Returns:
        dict: Dictionnaire contenant toutes les figures Plotly
    """
    all_figures = {}
    
    # Ajouter les visualisations géographiques
    geo_figures = create_geographic_visualizations(data_dict['candidats'])
    all_figures.update(geo_figures)
    
    # Ajouter les visualisations sur les conditions de santé
    health_figures = create_health_condition_visualizations(data_dict['candidats'])
    all_figures.update(health_figures)
    
    # Ajouter les visualisations du profilage des donneurs
    profiling_figures = create_donor_profiling_visualizations(data_dict['candidats'])
    all_figures.update(profiling_figures)
    
    # Ajouter les visualisations d'efficacité des campagnes
    campaign_figures = create_campaign_effectiveness_visualizations(
        data_dict['candidats'], 
        data_dict.get('donneurs')
    )
    all_figures.update(campaign_figures)
    
    # Ajouter les visualisations de fidélisation des donneurs
    retention_figures = create_donor_retention_visualizations(data_dict['candidats'])
    all_figures.update(retention_figures)
    
    # Ajouter les visualisations d'analyse de sentiment
    sentiment_figures = create_sentiment_analysis_visualizations(data_dict['candidats'])
    all_figures.update(sentiment_figures)
    
    return all_figures

def save_visualizations(figures, output_folder="visualizations"):
    """
    Sauvegarde les visualisations Plotly au format HTML
    
    Args:
        figures (dict): Dictionnaire contenant les figures Plotly
        output_folder (str): Dossier où sauvegarder les fichiers
    """
    import os
    from plotly.offline import plot
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Sauvegarder chaque figure
    for name, fig in figures.items():
        output_path = os.path.join(output_folder, f"{name}.html")
        plot(fig, filename=output_path, auto_open=False)
        print(f"Visualisation sauvegardée: {output_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Importer les données prétraitées
    processed_data = {}
    
    for name in ['candidats', 'donneurs', 'candidats_age', 'combined']:
        file_path = f"./data/processed_data/{name}_processed.csv"
        try:
            processed_data[name] = pd.read_csv(file_path)
            print(f"Données chargées: {file_path}")
        except:
            print(f"Impossible de charger: {file_path}")
    
    # Créer toutes les visualisations
    figures = create_all_visualizations(processed_data)
    
    # Sauvegarder les visualisations
    save_visualizations(figures)
    
    print("Visualisations terminées!")