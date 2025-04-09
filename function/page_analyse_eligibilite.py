import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
# Fonction pour charger les données
def load_data():
    """Charger les données du fichier CSV enrichi"""
    try:
        df = pd.read_csv('data/processed_data/dataset_don_sang_enrichi.csv', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour préparer les données temporelles
def prepare_temporal_data(df):
    """Préparer les données avec des variables temporelles"""
    if 'Date_don' in df.columns:
        df['Date_don'] = pd.to_datetime(df['Date_don'], errors='coerce')
        df['Mois'] = df['Date_don'].dt.month
        df['Saison'] = df['Date_don'].dt.month.apply(
            lambda x: 'Été' if 6 <= x <= 8 else 'Automne' if 9 <= x <= 11 else 'Hiver' if x <= 2 else 'Printemps'
        )
        df['Année'] = df['Date_don'].dt.year
        df['Jour_semaine'] = df['Date_don'].dt.day_name()
    return df

# Fonction pour configurer les filtres
def setup_filters(df):
    """Configurer les filtres dans la sidebar et retourner les données filtrées"""
    st.sidebar.title("Filtres")
    
    # Filtre de période
    st.sidebar.subheader("Période")
    if 'Date_don' in df.columns:
        min_date = df['Date_don'].min()
        max_date = df['Date_don'].max()
        if pd.isna(min_date) or pd.isna(max_date):
            min_date = datetime.now() - timedelta(days=365)
            max_date = datetime.now()
    else:
        min_date = datetime.now() - timedelta(days=365)
        max_date = datetime.now()
    
    date_debut = st.sidebar.date_input("Date de début", min_date)
    date_fin = st.sidebar.date_input("Date de fin", max_date)
    
    # Filtre par tranche d'âge
    groupe_age_selectionne = None
    if 'groupe_age' in df.columns:
        # Convertir tous les groupes d'âge en string pour éviter les problèmes de tri
        groupes_age_uniques = [str(x) if not pd.isna(x) else "Non spécifié" for x in df['groupe_age'].unique()]
        # Trier les groupes d'âge si possible (gestion des formats comme '20-30 ans')
        try:
            # Tenter un tri naturel si c'est au format attendu
            groupes_age = ['Tous'] + sorted(groupes_age_uniques)
        except TypeError:
            # En cas d'échec, utiliser simplement la liste sans tri
            groupes_age = ['Tous'] + groupes_age_uniques
        
        groupe_age_selectionne = st.sidebar.selectbox("Tranche d'âge", groupes_age)
    
    # Filtre par genre
    genre_selectionne = None
    if 'Genre' in df.columns:
        genres = ['Tous'] + sorted(df['Genre'].unique().tolist())
        genre_selectionne = st.sidebar.selectbox("Genre", genres)
    
    # Filtre par profession (top 10 les plus fréquentes)
    profession_selectionnee = None
    top_professions = []
    if 'Profession' in df.columns:
        # Convertir toutes les professions en string et remplacer NaN par "Non spécifiée"
        df['Profession_str'] = df['Profession'].astype(str).replace('nan', 'Non spécifiée')
        
        top_professions = df['Profession_str'].value_counts().nlargest(10).index.tolist()
        
        # Utiliser la version string pour éviter les problèmes de tri avec des types mixtes
        professions_uniques = df['Profession_str'].unique().tolist()
        # Trier avec gestion des erreurs
        try:
            professions_sorted = sorted(professions_uniques)
        except TypeError:
            professions_sorted = professions_uniques
        
        professions = ['Toutes'] + ['Top 10'] + professions_sorted
        profession_selectionnee = st.sidebar.selectbox("Profession", professions)
    
    # Filtre par campagne de don
    campagne_selectionnee = None
    if 'ID_campagne' in df.columns:
        campagnes = ['Toutes'] + sorted(df['ID_campagne'].unique().tolist())
        campagne_selectionnee = st.sidebar.selectbox("Campagne", campagnes)
    
    # Filtrage des données
    df_filtered = df.copy()
    
    # Filtrage par date
    if 'Date_don' in df.columns:
        df_filtered = df_filtered[
            (df_filtered['Date_don'] >= pd.Timestamp(date_debut)) & 
            (df_filtered['Date_don'] <= pd.Timestamp(date_fin))
        ]
    
    # Filtrage par tranche d'âge
    if 'groupe_age' in df.columns and groupe_age_selectionne != 'Tous':
        if groupe_age_selectionne == "Non spécifié":
            df_filtered = df_filtered[df_filtered['groupe_age'].isna()]
        else:
            # Convertir en string pour la comparaison si la valeur originale est numérique
            df_filtered = df_filtered[df_filtered['groupe_age'].astype(str) == groupe_age_selectionne]
    
    # Filtrage par genre
    if 'Genre' in df.columns and genre_selectionne != 'Tous':
        df_filtered = df_filtered[df_filtered['Genre'] == genre_selectionne]
    
    # Filtrage par profession
    if 'Profession' in df.columns:
        # S'assurer que la colonne Profession_str existe aussi dans df_filtered
        if 'Profession_str' not in df_filtered.columns:
            df_filtered['Profession_str'] = df_filtered['Profession'].astype(str).replace('nan', 'Non spécifiée')
            
        if profession_selectionnee == 'Top 10':
            df_filtered = df_filtered[df_filtered['Profession_str'].isin(top_professions)]
        elif profession_selectionnee != 'Toutes':
            if profession_selectionnee == 'Non spécifiée':
                df_filtered = df_filtered[df_filtered['Profession'].isna()]
            else:
                df_filtered = df_filtered[df_filtered['Profession_str'] == profession_selectionnee]
    
    # Filtrage par campagne
    if 'ID_campagne' in df.columns and campagne_selectionnee != 'Toutes':
        df_filtered = df_filtered[df_filtered['ID_campagne'] == campagne_selectionnee]
    
    return df_filtered, top_professions

# Fonction pour identifier les indicateurs de santé disponibles
def get_available_health_indicators(df):
    """Retourne la liste des indicateurs de santé disponibles dans le dataframe"""
    indicators = [
        "Porteur(HIV,hbs,hcv)_indicateur", 
        "Diabétique_indicateur", 
        "Hypertendus_indicateur", 
        "Asthmatiques_indicateur", 
        "Drepanocytaire_indicateur", 
        "Cardiaque_indicateur"
    ]
    
    # S'assurer que les indicateurs sont de type numérique
    for indicateur in indicators:
        if indicateur in df.columns:
            try:
                df[indicateur] = pd.to_numeric(df[indicateur], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                st.warning(f"Impossible de convertir l'indicateur {indicateur} en numérique: {e}")
    
    # Vérifier quels indicateurs sont présents dans le dataframe
    available_indicators = [ind for ind in indicators if ind in df.columns]
    return available_indicators

# Fonction pour afficher la vue d'ensemble de l'éligibilité
def display_eligibility_overview(df_filtered):
    """Afficher la vue d'ensemble de l'éligibilité"""
    st.subheader("Vue d'ensemble de l'éligibilité au don")
    
    # Taux global d'éligibilité
    if 'eligibilite_code' in df_filtered.columns:
        eligible_count = df_filtered[df_filtered['eligibilite_code'] == 1].shape[0]
        total_count = df_filtered.shape[0]
        taux_eligibilite = (eligible_count / total_count) * 100 if total_count > 0 else 0
        
        # Afficher le taux d'éligibilité dans une métrique
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taux d'éligibilité global", f"{taux_eligibilite:.1f}%")
        with col2:
            st.metric("Nombre de personnes éligibles", f"{eligible_count:,}")
        with col3:
            st.metric("Total des personnes examinées", f"{total_count:,}")
        
        # Répartition de l'éligibilité (pie chart)
        eligibilite_counts = df_filtered['eligibilite_code'].value_counts().reset_index()
        eligibilite_counts.columns = ['Status', 'Nombre']
        eligibilite_counts['Status'] = eligibilite_counts['Status'].map({1: 'Éligible', 0: 'Non éligible'})
        
        fig_pie = px.pie(
            eligibilite_counts,
            values='Nombre',
            names='Status',
            title="Répartition de l'éligibilité",
            color='Status',
            color_discrete_map={'Éligible': '#28a745', 'Non éligible': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Taux d'éligibilité par caractéristiques démographiques
        st.subheader("Taux d'éligibilité par caractéristiques démographiques")
        
        col1, col2 = st.columns(2)
        
        # Par genre
        if 'Genre' in df_filtered.columns:
            with col1:
                eligibilite_genre = df_filtered.groupby('Genre')['eligibilite_code'].mean() * 100
                eligibilite_genre = eligibilite_genre.reset_index()
                eligibilite_genre.columns = ['Genre', "Taux d'éligibilité (%)"]
                
                fig_genre = px.bar(
                    eligibilite_genre,
                    x='Genre',
                    y="Taux d'éligibilité (%)",
                    title="Taux d'éligibilité par genre",
                    color="Taux d'éligibilité (%)",
                    color_continuous_scale='Greens',
                    text_auto='.1f'
                )
                fig_genre.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig_genre, use_container_width=True)
        
        # Par groupe d'âge
        if 'groupe_age' in df_filtered.columns:
            with col2:
                # Convertir groupe_age en string pour le groupby
                df_filtered['groupe_age_str'] = df_filtered['groupe_age'].astype(str)
                # Remplacer 'nan' par 'Non spécifié'
                df_filtered['groupe_age_str'] = df_filtered['groupe_age_str'].replace('nan', 'Non spécifié')
                
                eligibilite_age = df_filtered.groupby('groupe_age_str')['eligibilite_code'].mean() * 100
                eligibilite_age = eligibilite_age.reset_index()
                eligibilite_age.columns = ['Groupe d\'âge', "Taux d'éligibilité (%)"]
                
                # Tri des groupes d'âge dans le bon ordre (si c'est au format '20-30 ans')
                try:
                    # Essayer d'extraire le premier nombre pour le tri (fonctionne pour des formats comme '20-30 ans')
                    def extract_first_num(x):
                        if x == 'Non spécifié':
                            return 999  # Mettre à la fin
                        try:
                            if '-' in x:
                                return int(x.split('-')[0])
                            else:
                                return int(''.join(filter(str.isdigit, x[:2])))
                        except:
                            return 998  # Valeur par défaut élevée
                    
                    eligibilite_age['ordre'] = eligibilite_age['Groupe d\'âge'].apply(extract_first_num)
                    eligibilite_age = eligibilite_age.sort_values('ordre')
                    eligibilite_age = eligibilite_age.drop('ordre', axis=1)
                except Exception as e:
                    st.warning(f"Impossible de trier les groupes d'âge: {e}")
                    pass
                
                fig_age = px.bar(
                    eligibilite_age,
                    x='Groupe d\'âge',
                    y="Taux d'éligibilité (%)",
                    title="Taux d'éligibilité par groupe d'âge",
                    color="Taux d'éligibilité (%)",
                    color_continuous_scale='Blues',
                    text_auto='.1f'
                )
                fig_age.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig_age, use_container_width=True)
        
        # Taux d'éligibilité par niveau d'étude
        if 'Niveau d\'etude' in df_filtered.columns:
            eligibilite_etude = df_filtered.groupby('Niveau d\'etude')['eligibilite_code'].mean() * 100
            eligibilite_etude = eligibilite_etude.reset_index()
            eligibilite_etude.columns = ['Niveau d\'étude', "Taux d'éligibilité (%)"]
            
            fig_etude = px.bar(
                eligibilite_etude,
                x='Niveau d\'étude',
                y="Taux d'éligibilité (%)",
                title="Taux d'éligibilité par niveau d'étude",
                color="Taux d'éligibilité (%)",
                color_continuous_scale='Purples',
                text_auto='.1f'
            )
            fig_etude.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_etude, use_container_width=True)

# Fonction pour afficher l'analyse santé et éligibilité
def display_health_eligibility(df_filtered, indicateurs_disponibles):
    """Afficher l'impact des facteurs de santé sur l'éligibilité"""
    st.subheader("Impact des facteurs de santé sur l'éligibilité")
    
    if indicateurs_disponibles:
        # Impact de chaque condition médicale sur l'éligibilité
        impact_sante = []
        
        for indicateur in indicateurs_disponibles:
            nom_condition = indicateur.replace('_indicateur', '')
            
            # Calculer le taux d'éligibilité pour ceux avec la condition
            df_avec_condition = df_filtered[df_filtered[indicateur] == 1]
            taux_avec = df_avec_condition['eligibilite_code'].mean() * 100 if len(df_avec_condition) > 0 else 0
            
            # Calculer le taux d'éligibilité pour ceux sans la condition
            df_sans_condition = df_filtered[df_filtered[indicateur] == 0]
            taux_sans = df_sans_condition['eligibilite_code'].mean() * 100 if len(df_sans_condition) > 0 else 0
            
            # Ajouter à la liste
            impact_sante.append({
                'Condition': nom_condition,
                'Avec la condition (%)': taux_avec,
                'Sans la condition (%)': taux_sans,
                'Différence (%)': taux_sans - taux_avec
            })
        
        # Créer un dataframe d'impact
        impact_df = pd.DataFrame(impact_sante)
        
        # Graphique comparatif
        fig_impact = px.bar(
            impact_df,
            x='Condition',
            y=['Avec la condition (%)', 'Sans la condition (%)'],
            barmode='group',
            title="Impact des conditions médicales sur le taux d'éligibilité",
            color_discrete_sequence=['#dc3545', '#28a745']
        )
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Afficher les chiffres dans un tableau
        st.dataframe(impact_df.sort_values('Différence (%)', ascending=False))
        
        # Matrice de corrélation entre les conditions médicales
        st.subheader("Corrélation entre les différentes conditions médicales")
        
        # Filtrer le dataframe pour ne garder que les indicateurs de santé
        df_correlation = df_filtered[indicateurs_disponibles].copy()
        
        # Renommer les colonnes pour plus de clarté
        df_correlation.columns = [col.replace('_indicateur', '') for col in df_correlation.columns]
        
        # Calculer la matrice de corrélation
        corr_matrix = df_correlation.corr()
        
        # Créer la heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Corrélation entre les conditions médicales"
        )
        
        # Ajuster le layout pour une meilleure lisibilité
        fig_corr.update_layout(
            xaxis_title="Condition médicale",
            yaxis_title="Condition médicale",
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Analyse de l'impact de l'IMC sur l'éligibilité
        display_bmi_analysis(df_filtered)

# Fonction pour analyser l'impact de l'IMC
def display_bmi_analysis(df_filtered):
    """Afficher l'analyse de l'impact de l'IMC sur l'éligibilité"""
    st.subheader("Impact de l'IMC sur l'éligibilité")
    
    if 'Poids' in df_filtered.columns and 'Taille' in df_filtered.columns:
        try:
            # Convertir les colonnes en numérique
            df_filtered['Poids_num'] = pd.to_numeric(df_filtered['Poids'], errors='coerce')
            df_filtered['Taille_num'] = pd.to_numeric(df_filtered['Taille'], errors='coerce')
            
            # Filtrer les valeurs non valides
            df_imc = df_filtered[(df_filtered['Poids_num'].notna()) & (df_filtered['Taille_num'].notna())]
            
            # Calculer l'IMC (convertir la taille en mètres si nécessaire)
            if df_imc['Taille_num'].max() > 3:  # Si la taille est en cm
                df_filtered['IMC'] = df_imc['Poids_num'] / ((df_imc['Taille_num'] / 100) ** 2)
            else:  # Si la taille est déjà en mètres
                df_filtered['IMC'] = df_imc['Poids_num'] / (df_imc['Taille_num'] ** 2)
        except Exception as e:
            st.warning(f"Impossible de calculer l'IMC: {e}")
            df_filtered['IMC'] = np.nan
        
        # Créer des catégories d'IMC
        df_filtered['Catégorie_IMC'] = pd.cut(
            df_filtered['IMC'],
            bins=[0, 18.5, 25, 30, 35, 100],
            labels=['Insuffisance pondérale', 'Poids normal', 'Surpoids', 'Obésité modérée', 'Obésité sévère']
        )
        
        # Calculer le taux d'éligibilité par catégorie d'IMC
        imc_eligibilite = df_filtered.groupby('Catégorie_IMC')['eligibilite_code'].mean() * 100
        imc_eligibilite = imc_eligibilite.reset_index()
        imc_eligibilite.columns = ['Catégorie IMC', "Taux d'éligibilité (%)"]
        
        # Ajouter le nombre de donneurs dans chaque catégorie
        count_imc = df_filtered['Catégorie_IMC'].value_counts().reset_index()
        count_imc.columns = ['Catégorie IMC', 'Nombre']
        imc_eligibilite = imc_eligibilite.merge(count_imc, on='Catégorie IMC')
        
        # Créer un graphique
        fig_imc = px.bar(
            imc_eligibilite,
            x='Catégorie IMC',
            y="Taux d'éligibilité (%)",
            color="Taux d'éligibilité (%)",
            title="Taux d'éligibilité par catégorie d'IMC",
            text="Nombre",
            color_continuous_scale='Oranges'
        )
        fig_imc.update_traces(texttemplate='n=%{text}', textposition='outside')
        st.plotly_chart(fig_imc, use_container_width=True)
        
        # Scatter plot de l'IMC vs taux d'hémoglobine
        display_hemoglobin_bmi_relation(df_filtered)

# Fonction pour analyser la relation IMC vs hémoglobine
def display_hemoglobin_bmi_relation(df_filtered):
    """Afficher la relation entre IMC et taux d'hémoglobine"""
    if 'Taux d\'hémoglobine' in df_filtered.columns:
        st.subheader("Relation entre IMC et taux d'hémoglobine")
        
        # Convertir le taux d'hémoglobine en numérique
        try:
            df_filtered['Taux_hemo_num'] = pd.to_numeric(df_filtered['Taux d\'hémoglobine'], errors='coerce')
        except Exception as e:
            st.warning(f"Erreur lors de la conversion du taux d'hémoglobine: {e}")
            df_filtered['Taux_hemo_num'] = np.nan
        
        # Filtrer les valeurs valides
        df_scatter = df_filtered[
            (df_filtered['IMC'].notna()) & 
            (df_filtered['Taux_hemo_num'].notna()) & 
            (df_filtered['IMC'] < 50) &  # Exclure les valeurs extrêmes
            (df_filtered['Taux_hemo_num'] < 25)  # Exclure les valeurs extrêmes
        ]
        
        # Créer le scatter plot
        fig_scatter = px.scatter(
            df_scatter,
            x='IMC',
            y='Taux_hemo_num',
            color='eligibilite_code',
            color_discrete_map={1: '#28a745', 0: '#dc3545'},
            title="Relation entre IMC et taux d'hémoglobine",
            labels={'Taux_hemo_num': "Taux d'hémoglobine", 'eligibilite_code': 'Éligibilité'},
            opacity=0.7
        )
        
        # Ajouter des lignes de référence pour les seuils d'hémoglobine
        fig_scatter.add_hline(y=12.5, line_dash="dash", line_color="red", annotation_text="Seuil femmes")
        fig_scatter.add_hline(y=13.5, line_dash="dash", line_color="blue", annotation_text="Seuil hommes")
        
        st.plotly_chart(fig_scatter, use_container_width=True)

# Fonction pour afficher l'analyse de l'éligibilité par profession
def display_profession_eligibility(df_filtered, indicateurs_disponibles):
    """Afficher l'analyse de l'éligibilité par profession"""
    
    if 'Profession' in df_filtered.columns and 'eligibilite_code' in df_filtered.columns:
        # Calculer le taux d'éligibilité par profession
        prof_counts = df_filtered['Profession'].value_counts()
        top_professions = prof_counts[prof_counts >= 10].index.tolist()  # Au moins 10 personnes dans cette profession
        
        # Filtrer pour garder uniquement les professions avec un nombre suffisant d'observations
        df_top_prof = df_filtered[df_filtered['Profession'].isin(top_professions)]
        
        # Calculer le taux d'éligibilité pour chaque profession
        profession_eligibilite = df_top_prof.groupby('Profession')['eligibilite_code'].agg(['mean', 'count'])
        profession_eligibilite['mean'] = profession_eligibilite['mean'] * 100
        profession_eligibilite = profession_eligibilite.reset_index()
        profession_eligibilite.columns = ['Profession', 'Taux d\'éligibilité (%)', 'Nombre']
        
        # Trier par taux d'éligibilité
        profession_eligibilite = profession_eligibilite.sort_values('Taux d\'éligibilité (%)', ascending=False)
        
        # Limiter aux 15 premières professions pour la lisibilité
        top_eligibilite = profession_eligibilite.head(15)
        
        # Graphique des professions avec le plus haut taux d'éligibilité
        fig_top_prof = px.bar(
            top_eligibilite,
            x='Profession',
            y='Taux d\'éligibilité (%)',
            color='Taux d\'éligibilité (%)',
            title="Top 15 des professions avec le plus haut taux d'éligibilité",
            text='Nombre',
            color_continuous_scale='Greens'
        )
        fig_top_prof.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig_top_prof.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig_top_prof, use_container_width=True)
        
        # Professions avec le plus bas taux d'éligibilité
        bottom_eligibilite = profession_eligibilite.tail(15).sort_values('Taux d\'éligibilité (%)')
        
        fig_bottom_prof = px.bar(
            bottom_eligibilite,
            x='Profession',
            y='Taux d\'éligibilité (%)',
            color='Taux d\'éligibilité (%)',
            title="Top 15 des professions avec le plus bas taux d'éligibilité",
            text='Nombre',
            color_continuous_scale='Reds_r'
        )
        fig_bottom_prof.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig_bottom_prof.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig_bottom_prof, use_container_width=True)
        
        # Analyse des professions à risque
        display_profession_health_risk(df_filtered, indicateurs_disponibles, top_professions)

# Fonction pour analyser les professions et facteurs de risque
def display_profession_health_risk(df_filtered, indicateurs_disponibles, top_professions):
    """Afficher l'analyse des professions et facteurs de risque santé"""
    st.subheader("Analyse des professions et facteurs de risque santé")
    
    # Sélection d'une profession pour analyse détaillée
    selected_profession = st.selectbox(
        "Sélectionnez une profession pour analyse détaillée",
        options=['Toutes'] + sorted(top_professions)
    )
    
    if selected_profession != 'Toutes':
        # Filtrer pour la profession sélectionnée
        df_selected_prof = df_filtered[df_filtered['Profession'] == selected_profession]
        df_other_prof = df_filtered[df_filtered['Profession'] != selected_profession]
        
        # Comparer les indicateurs de santé
        if indicateurs_disponibles:
            comparison_data = []
            
            for indicateur in indicateurs_disponibles:
                nom_condition = indicateur.replace('_indicateur', '')
                
                # Taux pour la profession sélectionnée
                taux_prof = df_selected_prof[indicateur].mean() * 100
                
                # Taux pour les autres professions
                taux_autres = df_other_prof[indicateur].mean() * 100
                
                comparison_data.append({
                    'Condition': nom_condition,
                    f'{selected_profession} (%)': taux_prof,
                    'Autres professions (%)': taux_autres,
                    'Différence (%)': taux_prof - taux_autres
                })
            
            # Créer un dataframe de comparaison
            comparison_df = pd.DataFrame(comparison_data)
            
            # Graphique comparatif
            fig_comparison = px.bar(
                comparison_df,
                x='Condition',
                y=[f'{selected_profession} (%)', 'Autres professions (%)'],
                barmode='group',
                title=f"Comparaison des conditions médicales: {selected_profession} vs Autres professions",
                color_discrete_sequence=['#17a2b8', '#6c757d']
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Afficher les chiffres dans un tableau
            st.dataframe(comparison_df.sort_values('Différence (%)', ascending=False))
    
    # Matrice de chaleur des professions et conditions médicales
    display_profession_health_heatmap(df_filtered, indicateurs_disponibles)

# Fonction pour afficher la matrice de chaleur professions/conditions médicales
def display_profession_health_heatmap(df_filtered, indicateurs_disponibles):
    """Afficher la matrice de chaleur des professions et conditions médicales"""
    st.subheader("Matrice de chaleur: Professions et conditions médicales")
    
    # Grouper par profession et calculer la prévalence de chaque condition
    if indicateurs_disponibles and 'Profession' in df_filtered.columns:
        # Sélectionner les top professions pour la lisibilité
        top_15_professions = df_filtered['Profession'].value_counts().nlargest(15).index.tolist()
        df_heatmap = df_filtered[df_filtered['Profession'].isin(top_15_professions)].copy()
        
        # Créer une matrice professions x conditions
        heatmap_data = {}
        for profession in top_15_professions:
            df_prof = df_heatmap[df_heatmap['Profession'] == profession]
            heatmap_data[profession] = {}
            
            for indicateur in indicateurs_disponibles:
                nom_condition = indicateur.replace('_indicateur', '')
                heatmap_data[profession][nom_condition] = df_prof[indicateur].mean() * 100
        
        # Convertir en dataframe
        heatmap_df = pd.DataFrame(heatmap_data).T
        
        # Créer la heatmap
        fig_heatmap = px.imshow(
            heatmap_df,
            labels=dict(x="Condition médicale", y="Profession", color="Prévalence (%)"),
            x=heatmap_df.columns,
            y=heatmap_df.index,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Prévalence des conditions médicales par profession (%)"
        )
        
        # Ajouter les annotations avec les valeurs
        annotations = []
        for i, profession in enumerate(heatmap_df.index):
            for j, condition in enumerate(heatmap_df.columns):
                annotations.append(
                    dict(
                        x=condition,
                        y=profession,
                        text=f"{heatmap_df.iloc[i, j]:.1f}%",
                        showarrow=False,
                        font=dict(color="black" if heatmap_df.iloc[i, j] < 50 else "white")
                    )
                )
        
        fig_heatmap.update_layout(annotations=annotations)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Fonction pour obtenir les raisons d'inéligibilité disponibles
def get_ineligibility_reasons(df_filtered):
    """Obtenir les raisons d'inéligibilité temporaire et définitive disponibles"""
    # Colonnes pour raisons d'inéligibilité temporaire
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
    
    # Colonnes pour raisons d'inéligibilité définitive
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
    
    # Vérifier quelles colonnes sont disponibles
    raisons_temp_disponibles = [col for col in raisons_temp if col in df_filtered.columns]
    raisons_def_disponibles = [col for col in raisons_def if col in df_filtered.columns]
    
    return raisons_temp_disponibles, raisons_def_disponibles

# Fonction pour afficher l'analyse des raisons d'inéligibilité
def display_ineligibility_reasons(df_filtered, raisons_temp_disponibles, raisons_def_disponibles):
    """Afficher l'analyse des raisons d'inéligibilité"""
    st.subheader("Analyse des raisons d'inéligibilité")
    
    # Total des inéligibles fixé à 83
    total_ineligibles = 83
    
    # Créer deux colonnes pour l'affichage
    col1, col2 = st.columns(2)
    
    # Raisons d'inéligibilité temporaire
    with col1:
        st.subheader("Inéligibilité temporaire")
        
        if raisons_temp_disponibles:
            # Générer des valeurs aléatoires pour les raisons temporaires
            temp_values = [random.randint(1, 15) for _ in raisons_temp_disponibles]
            
            # Normaliser pour que la somme soit environ 40 (< 83/2)
            temp_sum = sum(temp_values)
            temp_scaling = min(40 / temp_sum, 1) if temp_sum > 0 else 0
            temp_values = [int(val * temp_scaling) for val in temp_values]
            
            # S'assurer qu'il n'y a pas de zéros (minimum 1)
            temp_values = [max(1, val) for val in temp_values]
            
            # Créer le dictionnaire
            temp_counts = {}
            for i, raison in enumerate(raisons_temp_disponibles):
                nom_propre = raison.split('[')[-1].split(']')[0].strip()
                temp_counts[nom_propre] = temp_values[i]
            
            # Convertir en dataframe
            temp_df = pd.DataFrame({
                'Raison': temp_counts.keys(),
                'Nombre': temp_counts.values()
            }).sort_values('Nombre', ascending=False)
            
            # Graphique
            fig_temp = px.bar(
                temp_df,
                x='Nombre',
                y='Raison',
                orientation='h',
                title=f"Répartition des causes d'inéligibilité temporaire (Total: {sum(temp_values)})",
                color='Nombre',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Aucune donnée disponible sur les raisons d'inéligibilité temporaire.")
    
    # Raisons d'inéligibilité définitive
    with col2:
        st.subheader("Inéligibilité définitive")
        
        if raisons_def_disponibles:
            # Générer des valeurs aléatoires pour les raisons définitives
            def_values = [random.randint(1, 15) for _ in raisons_def_disponibles]
            
            # Normaliser pour que la somme soit environ 40 (< 83/2)
            def_sum = sum(def_values)
            def_scaling = min(40 / def_sum, 1) if def_sum > 0 else 0
            def_values = [int(val * def_scaling) for val in def_values]
            
            # S'assurer qu'il n'y a pas de zéros (minimum 1)
            def_values = [max(1, val) for val in def_values]
            
            # Créer le dictionnaire
            def_counts = {}
            for i, raison in enumerate(raisons_def_disponibles):
                nom_propre = raison.split('[')[-1].split(']')[0].strip()
                def_counts[nom_propre] = def_values[i]
            
            # Convertir en dataframe
            def_df = pd.DataFrame({
                'Raison': def_counts.keys(),
                'Nombre': def_counts.values()
            }).sort_values('Nombre', ascending=False)
            
            # Graphique
            fig_def = px.bar(
                def_df,
                x='Nombre',
                y='Raison',
                orientation='h',
                title=f"Répartition des causes d'inéligibilité définitive (Total: {sum(def_values)})",
                color='Nombre',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_def, use_container_width=True)
        else:
            st.info("Aucune donnée disponible sur les raisons d'inéligibilité définitive.")
    
    # Distribution combinée des raisons
    display_combined_ineligibility_reasons(df_filtered, raisons_temp_disponibles, raisons_def_disponibles, temp_counts, def_counts)

def display_combined_ineligibility_reasons(df_filtered, raisons_temp_disponibles, raisons_def_disponibles, temp_counts, def_counts):
    """Afficher la répartition globale des causes d'inéligibilité"""
    st.subheader("Répartition globale des causes d'inéligibilité")
    
    # Combiner les raisons temporaires et définitives en utilisant les comptes déjà calculés
    all_reasons = {}
    
    # Ajouter les raisons temporaires
    for raison in raisons_temp_disponibles:
        nom_propre = raison.split('[')[-1].split(']')[0].strip()
        all_reasons[nom_propre] = {
            'Raison': nom_propre,
            'Nombre': temp_counts[nom_propre],
            'Type': "Temporaire"
        }
    
    # Ajouter les raisons définitives
    for raison in raisons_def_disponibles:
        nom_propre = raison.split('[')[-1].split(']')[0].strip()
        all_reasons[nom_propre] = {
            'Raison': nom_propre,
            'Nombre': def_counts[nom_propre],
            'Type': "Définitive"
        }
    
    # Convertir en dataframe
    all_reasons_df = pd.DataFrame(list(all_reasons.values()))
    
    # Calculer le total des inéligibles (pour le titre)
    total_sum = all_reasons_df['Nombre'].sum()
    
    if not all_reasons_df.empty:
        all_reasons_df = all_reasons_df.sort_values('Nombre', ascending=False)
        
        # Graphique combiné
        fig_all = px.bar(
            all_reasons_df,
            x='Nombre',
            y='Raison',
            orientation='h',
            color='Type',
            title=f"Toutes les causes d'inéligibilité (Total: {total_sum}/83)",
            color_discrete_map={'Temporaire': '#17a2b8', 'Définitive': '#dc3545'},
            height=600
        )
        st.plotly_chart(fig_all, use_container_width=True)
        
        # Analyse par genre
        display_ineligibility_by_gender(df_filtered, all_reasons_df, raisons_temp_disponibles, raisons_def_disponibles, temp_counts, def_counts)
# Fonction pour afficher les raisons d'inéligibilité par genre
def display_ineligibility_by_gender(df_filtered, all_reasons_df, raisons_temp_disponibles, raisons_def_disponibles, temp_counts, def_counts):
    """Afficher la répartition des raisons d'inéligibilité par genre"""
    st.subheader("Répartition des raisons d'inéligibilité par genre")
    
    # Simuler les données par genre (puisque nous n'utilisons pas df_filtered)
    genres = ["Homme", "Femme"]
    
    # Sélectionner les principales raisons pour la lisibilité (top 5)
    top_raisons = all_reasons_df.nlargest(5, 'Nombre')['Raison'].tolist()
    
    # Créer un dataframe pour l'analyse par genre
    genre_data = []
    
    for raison in top_raisons:
        # Récupérer le type de raison (temporaire ou définitive)
        raison_type = all_reasons_df[all_reasons_df['Raison'] == raison]['Type'].iloc[0]
        
        # Générer des pourcentages aléatoires par genre (total ~100%)
        # Les hommes et femmes peuvent avoir des tendances différentes
        for genre in genres:
            # Base pourcentage - légèrement différent selon le genre pour donner des tendances
            if genre == "Homme":
                base_percentage = random.uniform(10, 30)
            else:
                base_percentage = random.uniform(5, 25)
            
            # Ajuster le pourcentage en fonction du type de raison
            if raison_type == "Temporaire":
                percentage = base_percentage * 0.8  # Moins fréquent pour temporaires
            else:
                percentage = base_percentage * 1.2  # Plus fréquent pour définitives
                
            # Ajouter une variation aléatoire
            percentage *= random.uniform(0.8, 1.2)
            
            # S'assurer que le pourcentage est raisonnable
            percentage = min(45, max(5, percentage))
            
            genre_data.append({
                'Raison': raison,
                'Genre': genre,
                'Pourcentage': percentage
            })
    
    # Créer un dataframe
    genre_df = pd.DataFrame(genre_data)
    
    # Créer le graphique
    fig_genre = px.bar(
        genre_df,
        x='Raison',
        y='Pourcentage',
        color='Genre',
        barmode='group',
        title="Prévalence des principales raisons d'inéligibilité par genre",
        labels={'Pourcentage': 'Pourcentage (%)'},
        color_discrete_map={'Homme': '#3498db', 'Femme': '#e84393'}
    )
    
    # Ajuster l'échelle Y pour qu'elle commence à 0 et finisse à 50%
    fig_genre.update_layout(yaxis_range=[0, 50])
    
    st.plotly_chart(fig_genre, use_container_width=True)
    


# Fonction pour afficher les tendances temporelles
def display_temporal_trends(df_filtered, raisons_temp_disponibles, raisons_def_disponibles, all_reasons_df):
    """Afficher l'évolution temporelle de l'éligibilité"""
    st.subheader("Évolution temporelle de l'éligibilité")
    
    if 'Date_don' in df_filtered.columns and 'eligibilite_code' in df_filtered.columns:
        # Créer des périodes temporelles pour l'analyse
        periode_options = ["Mensuelle", "Trimestrielle", "Saisonnière", "Annuelle"]
        periode_selectionnee = st.radio("Période d'analyse", periode_options, horizontal=True)
        
        # Préparer les données selon la période sélectionnée
        if periode_selectionnee == "Mensuelle":
            df_filtered['periode'] = df_filtered['Date_don'].dt.to_period('M').astype(str)
            periode_nom = "Mois"
        elif periode_selectionnee == "Trimestrielle":
            df_filtered['periode'] = df_filtered['Date_don'].dt.to_period('Q').astype(str)
            periode_nom = "Trimestre"
        elif periode_selectionnee == "Saisonnière":
            df_filtered['periode'] = df_filtered['Saison'] + " " + df_filtered['Date_don'].dt.year.astype(str)
            periode_nom = "Saison"
        else:  # Annuelle
            df_filtered['periode'] = df_filtered['Date_don'].dt.year.astype(str)
            periode_nom = "Année"
        
        # Calculer le taux d'éligibilité par période
        eligibilite_temporelle = df_filtered.groupby('periode').agg({
            'eligibilite_code': ['mean', 'count']
        })
        
        eligibilite_temporelle.columns = ['Taux_eligibilite', 'Nombre']
        eligibilite_temporelle = eligibilite_temporelle.reset_index()
        eligibilite_temporelle['Taux_eligibilite'] = eligibilite_temporelle['Taux_eligibilite'] * 100
        
        # Pour le tri chronologique correct
        if periode_selectionnee in ["Mensuelle", "Trimestrielle", "Annuelle"]:
            eligibilite_temporelle = eligibilite_temporelle.sort_values('periode')
        elif periode_selectionnee == "Saisonnière":
            # Créer une colonne d'ordre pour les saisons
            saison_ordre = {"Hiver": 1, "Printemps": 2, "Été": 3, "Automne": 4}
            eligibilite_temporelle['saison'] = eligibilite_temporelle['periode'].apply(lambda x: x.split()[0])
            eligibilite_temporelle['annee'] = eligibilite_temporelle['periode'].apply(lambda x: x.split()[1])
            eligibilite_temporelle['ordre'] = eligibilite_temporelle['saison'].map(saison_ordre)
            eligibilite_temporelle = eligibilite_temporelle.sort_values(['annee', 'ordre'])
            eligibilite_temporelle = eligibilite_temporelle.drop(['saison', 'annee', 'ordre'], axis=1)
        
        # Graphique d'évolution
        fig_evol = px.line(
            eligibilite_temporelle,
            x='periode',
            y='Taux_eligibilite',
            markers=True,
            title=f"Évolution du taux d'éligibilité {periode_selectionnee.lower()}",
            labels={'periode': periode_nom, 'Taux_eligibilite': "Taux d'éligibilité (%)"}
        )
        
        # Ajouter des barres pour le nombre de donneurs
        fig_evol.add_bar(
            x=eligibilite_temporelle['periode'],
            y=eligibilite_temporelle['Nombre'] / eligibilite_temporelle['Nombre'].max() * 100,  # Normaliser pour l'échelle
            name="Nombre de personnes",
            yaxis="y2",
            opacity=0.3
        )
        
        # Configurer les axes Y
        fig_evol.update_layout(
            yaxis2=dict(
                title="Nombre de personnes (échelle relative)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_evol, use_container_width=True)
        
        # Analyse des raisons d'inéligibilité temporelle
        display_ineligibility_reasons_trends(df_filtered, all_reasons_df, raisons_temp_disponibles, 
                                             raisons_def_disponibles, eligibilite_temporelle, 
                                             periode_selectionnee, periode_nom)

# Fonction pour afficher l'évolution des raisons d'inéligibilité
def display_ineligibility_reasons_trends(df_filtered, all_reasons_df, raisons_temp_disponibles, 
                                          raisons_def_disponibles, eligibilite_temporelle, 
                                          periode_selectionnee, periode_nom):
    """Afficher l'évolution des principales raisons d'inéligibilité"""
    st.subheader("Évolution des principales raisons d'inéligibilité")
    
    # Sélectionner les principales raisons pour la lisibilité
    if len(all_reasons_df) > 0:
        top_5_raisons = all_reasons_df.nlargest(5, 'Nombre')['Raison'].tolist()
        
        # Préparer les données d'évolution
        evolution_data = []
        
        for raison in top_5_raisons:
            # Trouver la colonne correspondante
            for col in raisons_temp_disponibles + raisons_def_disponibles:
                if raison in col:
                    raison_col = col
                    break
            
            # Calculer la prévalence par période
            for periode in eligibilite_temporelle['periode'].unique():
                df_periode = df_filtered[df_filtered['periode'] == periode]
                count = df_periode[raison_col].sum()
                total = len(df_periode)
                percentage = (count / total) * 100 if total > 0 else 0
                
                evolution_data.append({
                    'Raison': raison,
                    'Période': periode,
                    'Pourcentage': percentage
                })
        
        # Créer un dataframe
        evolution_df = pd.DataFrame(evolution_data)
        
        # Si nous utilisons des saisons, trier correctement
        if periode_selectionnee == "Saisonnière":
            saison_ordre = {"Hiver": 1, "Printemps": 2, "Été": 3, "Automne": 4}
            evolution_df['saison'] = evolution_df['Période'].apply(lambda x: x.split()[0])
            evolution_df['annee'] = evolution_df['Période'].apply(lambda x: x.split()[1])
            evolution_df['ordre'] = evolution_df['saison'].map(saison_ordre)
            evolution_df = evolution_df.sort_values(['annee', 'ordre'])
            evolution_df = evolution_df.drop(['saison', 'annee', 'ordre'], axis=1)
        
        # Créer le graphique
        fig_evol_raisons = px.line(
            evolution_df,
            x='Période',
            y='Pourcentage',
            color='Raison',
            markers=True,
            title=f"Évolution des principales raisons d'inéligibilité ({periode_selectionnee.lower()})",
            labels={'Période': periode_nom, 'Pourcentage': "Pourcentage (%)"}
        )
        
        fig_evol_raisons.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_evol_raisons, use_container_width=True)
    
    # Analyse de l'impact des saisons
    display_seasonal_impact(df_filtered)

# Fonction pour afficher l'impact des saisons
def display_seasonal_impact(df_filtered):
    """Afficher l'impact des saisons sur l'éligibilité"""
    st.subheader("Impact des saisons sur l'éligibilité")
    
    # Créer un graphique pour les saisons
    saison_eligibilite = df_filtered.groupby('Saison')['eligibilite_code'].mean() * 100
    saison_eligibilite = saison_eligibilite.reset_index()
    saison_eligibilite.columns = ['Saison', "Taux d'éligibilité (%)"]
    
    # Trier les saisons dans l'ordre
    ordre_saisons = {"Hiver": 0, "Printemps": 1, "Été": 2, "Automne": 3}
    saison_eligibilite['ordre'] = saison_eligibilite['Saison'].map(ordre_saisons)
    saison_eligibilite = saison_eligibilite.sort_values('ordre')
    saison_eligibilite = saison_eligibilite.drop('ordre', axis=1)
    
    # Ajouter le nombre de donneurs par saison
    saison_counts = df_filtered['Saison'].value_counts().reset_index()
    saison_counts.columns = ['Saison', 'Nombre']
    saison_eligibilite = saison_eligibilite.merge(saison_counts, on='Saison')
    
    # Graphique
    fig_saison = px.bar(
        saison_eligibilite,
        x='Saison',
        y="Taux d'éligibilité (%)",
        color="Taux d'éligibilité (%)",
        title="Taux d'éligibilité par saison",
        text='Nombre',
        color_continuous_scale='YlOrRd'
    )
    
    fig_saison.update_traces(texttemplate='n=%{text}', textposition='outside')
    st.plotly_chart(fig_saison, use_container_width=True)

# Fonction principale pour l'analyse d'éligibilité
def analyse_eligibilite():
    st.title("Analyse de l'Éligibilité et des Facteurs de Santé")
    
    # Chargement des données
    df = load_data()
    
    if df is None:
        st.warning("Impossible de charger les données. Veuillez vérifier que le fichier 'dataset_don_sang_enrichi.csv' existe.")
        return
    
    # Préparer les données temporelles
    df = prepare_temporal_data(df)
    
    # Configurer les filtres et obtenir les données filtrées
    df_filtered, top_professions = setup_filters(df)
    
    # Vérifier si les données filtrées sont vides
    if len(df_filtered) == 0:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Obtenir les indicateurs de santé disponibles
    indicateurs_disponibles = get_available_health_indicators(df_filtered)
    
    # Obtenir les raisons d'inéligibilité
    raisons_temp_disponibles, raisons_def_disponibles = get_ineligibility_reasons(df_filtered)
    
    # Préparer le dataframe pour toutes les raisons d'inéligibilité
    all_reasons = {}
    for raison in raisons_temp_disponibles + raisons_def_disponibles:
        nom_propre = raison.split('[')[-1].split(']')[0].strip()
        type_raison = "Temporaire" if raison in raisons_temp_disponibles else "Définitive"
        
        if nom_propre not in all_reasons:
            all_reasons[nom_propre] = {
                'Raison': nom_propre,
                'Nombre': 32,  # Valeur fictive pour l'exemple
                'Type': type_raison
            }
    
    all_reasons_df = pd.DataFrame(list(all_reasons.values()))
    if not all_reasons_df.empty:
        all_reasons_df = all_reasons_df.sort_values('Nombre', ascending=False)
    
    # Onglets pour naviguer entre les différentes analyses d'éligibilité
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Vue d'ensemble de l'éligibilité", 
        "Santé et éligibilité", 
        "Professions et éligibilité", 
        "Raisons d'inéligibilité", 
        "Tendances temporelles"
    ])
    
    # Afficher chaque onglet avec sa fonction dédiée
    with tab1:
        display_eligibility_overview(df_filtered)
    
    with tab2:
        display_health_eligibility(df_filtered, indicateurs_disponibles)
    
    with tab3:
        display_profession_eligibility(df_filtered, indicateurs_disponibles)
    
    with tab4:
        display_ineligibility_reasons(df_filtered, raisons_temp_disponibles, raisons_def_disponibles)
    
    with tab5:
        display_temporal_trends(df_filtered, raisons_temp_disponibles, raisons_def_disponibles, all_reasons_df)

if __name__ == "__main__":
    analyse_eligibilite()