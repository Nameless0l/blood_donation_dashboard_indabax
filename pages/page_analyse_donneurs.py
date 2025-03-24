import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def load_data():
    """Charger les données du fichier CSV enrichi"""
    try:
        df = pd.read_csv('dataset_don_sang_enrichi.csv', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

def analyse_donneurs():
    st.title("Analyse du Profilage des Donneurs")
    
    # Chargement des données
    df = load_data()
    
    if df is None:
        st.warning("Impossible de charger les données. Veuillez vérifier que le fichier 'dataset_don_sang_enrichi.csv' existe.")
        return
    
    # Convertir les colonnes de date pour l'analyse temporelle
    date_columns = ['Date_don', 'Date de remplissage de la fiche', 'Si oui preciser la date du dernier don.']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Impossible de convertir la colonne {col} en date: {e}")
    
    # Sidebar pour les filtres
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
    if 'groupe_age' in df.columns:
        # Convertir tous les groupes d'âge en string pour éviter les problèmes de tri
        groupes_age_uniques = [str(x) if not pd.isna(x) else "Non spécifié" for x in df['groupe_age'].unique()]
        # Trier les groupes d'âge si possible
        try:
            groupes_age = ['Tous'] + sorted(groupes_age_uniques)
        except TypeError:
            groupes_age = ['Tous'] + groupes_age_uniques
        
        groupe_age_selectionne = st.sidebar.selectbox("Tranche d'âge", groupes_age)
    
    # Filtre par genre
    if 'Genre' in df.columns:
        genres = ['Tous'] + sorted(df['Genre'].unique().tolist())
        genre_selectionne = st.sidebar.selectbox("Genre", genres)
    
    # Filtre par type de donneur
    statut_donneur = st.sidebar.radio(
        "Statut du donneur",
        ['Tous', 'Primo-donneurs', 'Donneurs réguliers']
    )
    
    # Filtre par campagne de don
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
    
    # Filtrage par statut de donneur
    if 'experience_don' in df.columns and statut_donneur != 'Tous':
        if statut_donneur == 'Primo-donneurs':
            df_filtered = df_filtered[df_filtered['experience_don'] == 0]
        elif statut_donneur == 'Donneurs réguliers':
            df_filtered = df_filtered[df_filtered['experience_don'] == 1]
    
    # Filtrage par campagne
    if 'ID_campagne' in df.columns and campagne_selectionnee != 'Toutes':
        df_filtered = df_filtered[df_filtered['ID_campagne'] == campagne_selectionnee]
    
    # Vérifier si les données filtrées sont vides
    if len(df_filtered) == 0:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Onglets pour naviguer entre les différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Fréquence des dons", 
        "Communication efficace", 
        "Motivations des donneurs", 
        "Analyse des sentiments"
    ])
    
    # Tab 1: Fréquence des dons
    with tab1:
        st.subheader("Analyse de la fréquence des dons")
        
        # Calcul de la fréquence moyenne des dons
        if 'jours_depuis_dernier_don' in df_filtered.columns and 'experience_don' in df_filtered.columns:
            donneurs_reguliers = df_filtered[df_filtered['experience_don'] == 1]
            
            if len(donneurs_reguliers) > 0:
                # Convertir en numérique et gérer les valeurs manquantes
                donneurs_reguliers['jours_num'] = pd.to_numeric(donneurs_reguliers['jours_depuis_dernier_don'], errors='coerce')
                
                # Exclure les valeurs extrêmes (plus de 3 ans ou valeurs négatives)
                donneurs_valides = donneurs_reguliers[
                    (donneurs_reguliers['jours_num'] > 0) & 
                    (donneurs_reguliers['jours_num'] < 1095)  # ~3 ans
                ]
                
                if len(donneurs_valides) > 0:
                    jours_moyens = donneurs_valides['jours_num'].mean()
                    jours_medians = donneurs_valides['jours_num'].median()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Fréquence moyenne entre les dons", 
                            f"{jours_moyens:.1f} jours",
                            help="Nombre moyen de jours entre deux dons pour les donneurs réguliers"
                        )
                    
                    with col2:
                        st.metric(
                            "Fréquence médiane entre les dons", 
                            f"{jours_medians:.1f} jours",
                            help="Nombre médian de jours entre deux dons (valeur centrale)"
                        )
                    
                    with col3:
                        st.metric(
                            "Dons par donneur par an (estimé)", 
                            f"{(365/jours_moyens):.1f}",
                            help="Estimation du nombre moyen de dons par donneur sur une année"
                        )
                    
                    # Distribution de la fréquence des dons
                    fig_freq = px.histogram(
                        donneurs_valides,
                        x='jours_num',
                        nbins=20,
                        title="Distribution du nombre de jours entre les dons",
                        labels={'jours_num': 'Jours depuis le dernier don'},
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig_freq.update_layout(xaxis_title="Jours depuis le dernier don", yaxis_title="Nombre de donneurs")
                    st.plotly_chart(fig_freq, use_container_width=True)
                    
                    # Ajouter des statistiques supplémentaires
                    st.subheader("Répartition des donneurs par fréquence")
                    
                    # Catégoriser les donneurs par fréquence
                    donneurs_valides['categorie_frequence'] = pd.cut(
                        donneurs_valides['jours_num'],
                        bins=[0, 90, 180, 365, 730, 1095],
                        labels=['0-3 mois', '3-6 mois', '6-12 mois', '1-2 ans', '2-3 ans']
                    )
                    
                    # Compter les donneurs par catégorie
                    freq_counts = donneurs_valides['categorie_frequence'].value_counts().reset_index()
                    freq_counts.columns = ['Intervalle entre dons', 'Nombre de donneurs']
                    
                    # Graphique
                    fig_categories = px.pie(
                        freq_counts,
                        values='Nombre de donneurs',
                        names='Intervalle entre dons',
                        title="Répartition des donneurs par intervalle entre dons",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig_categories, use_container_width=True)
                else:
                    st.info("Pas assez de données valides sur la fréquence des dons.")
            else:
                st.info("Aucun donneur régulier dans les données filtrées.")
        else:
            st.info("Données insuffisantes pour analyser la fréquence des dons.")
        
        # Analyser la fidélité des donneurs
        st.subheader("Fidélité des donneurs")
        
        if 'experience_don' in df_filtered.columns:
            # Calculer le taux de fidélisation global
            taux_fidelisation = df_filtered['experience_don'].mean() * 100
            
            st.metric(
                "Taux de donneurs réguliers", 
                f"{taux_fidelisation:.1f}%",
                help="Pourcentage de donneurs ayant déjà donné leur sang auparavant"
            )
            
            # Analyser la fidélité par caractéristiques démographiques
            col1, col2 = st.columns(2)
            
            # Par genre
            if 'Genre' in df_filtered.columns:
                with col1:
                    fidelite_genre = df_filtered.groupby('Genre')['experience_don'].mean() * 100
                    fidelite_genre = fidelite_genre.reset_index()
                    fidelite_genre.columns = ['Genre', "Taux de fidélisation (%)"]
                    
                    fig_genre = px.bar(
                        fidelite_genre,
                        x='Genre',
                        y="Taux de fidélisation (%)",
                        title="Fidélisation par genre",
                        color="Taux de fidélisation (%)",
                        color_continuous_scale='Reds',
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
                    
                    fidelite_age = df_filtered.groupby('groupe_age_str')['experience_don'].mean() * 100
                    fidelite_age = fidelite_age.reset_index()
                    fidelite_age.columns = ['Groupe d\'âge', "Taux de fidélisation (%)"]
                    
                    # Essayer de trier les groupes d'âge
                    try:
                        def extract_first_num(x):
                            if x == 'Non spécifié':
                                return 999
                            try:
                                if '-' in x:
                                    return int(x.split('-')[0])
                                else:
                                    return int(''.join(filter(str.isdigit, x[:2])))
                            except:
                                return 998
                        
                        fidelite_age['ordre'] = fidelite_age['Groupe d\'âge'].apply(extract_first_num)
                        fidelite_age = fidelite_age.sort_values('ordre')
                        fidelite_age = fidelite_age.drop('ordre', axis=1)
                    except Exception as e:
                        st.warning(f"Impossible de trier les groupes d'âge: {e}")
                    
                    fig_age = px.bar(
                        fidelite_age,
                        x='Groupe d\'âge',
                        y="Taux de fidélisation (%)",
                        title="Fidélisation par groupe d'âge",
                        color="Taux de fidélisation (%)",
                        color_continuous_scale='Blues',
                        text_auto='.1f'
                    )
                    fig_age.update_traces(texttemplate='%{text}%', textposition='outside')
                    st.plotly_chart(fig_age, use_container_width=True)
            
            # Préférence pour les dons futurs
            if 'Fréquence_don_souhaitée' in df_filtered.columns:
                st.subheader("Fréquence de don souhaitée pour l'avenir")
                
                # Convertir en catégorie et compter
                freq_souhaitee = df_filtered['Fréquence_don_souhaitée'].value_counts().reset_index()
                freq_souhaitee.columns = ['Fréquence souhaitée', 'Nombre de donneurs']
                
                # Graphique
                fig_souhaitee = px.bar(
                    freq_souhaitee,
                    x='Fréquence souhaitée',
                    y='Nombre de donneurs',
                    title="Fréquence de don souhaitée par les donneurs",
                    color='Nombre de donneurs',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_souhaitee, use_container_width=True)
        else:
            st.info("Données insuffisantes pour analyser la fidélité des donneurs.")
    
    # Tab 2: Communication efficace
    with tab2:
        st.subheader("Analyse des canaux de communication efficaces")
        
        # Analyser comment les donneurs ont été informés
        if 'Comment_informé' in df_filtered.columns:
            # Compter les occurrences de chaque canal
            canal_counts = df_filtered['Comment_informé'].value_counts().reset_index()
            canal_counts.columns = ['Canal d\'information', 'Nombre de donneurs']
            
            # Trier pour avoir les plus efficaces en premier
            canal_counts = canal_counts.sort_values('Nombre de donneurs', ascending=False)
            
            # Graphique des canaux d'information
            fig_canaux = px.bar(
                canal_counts,
                x='Canal d\'information',
                y='Nombre de donneurs',
                title="Comment les donneurs ont été informés de la campagne",
                color='Nombre de donneurs',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_canaux, use_container_width=True)
            
            # Analyser les canaux spécifiques
            if 'Canal_information' in df_filtered.columns:
                # Compter les occurrences de chaque canal spécifique
                canal_spec_counts = df_filtered['Canal_information'].value_counts().reset_index()
                canal_spec_counts.columns = ['Canal spécifique', 'Nombre de donneurs']
                
                # Trier pour avoir les plus efficaces en premier
                canal_spec_counts = canal_spec_counts.sort_values('Nombre de donneurs', ascending=False)
                
                # Graphique des canaux spécifiques
                fig_canaux_spec = px.bar(
                    canal_spec_counts,
                    x='Canal spécifique',
                    y='Nombre de donneurs',
                    title="Canaux de communication spécifiques utilisés",
                    color='Nombre de donneurs',
                    color_continuous_scale='Turbo'
                )
                st.plotly_chart(fig_canaux_spec, use_container_width=True)
            
            # Analyser l'efficacité des canaux pour les primo-donneurs vs donneurs réguliers
            if 'experience_don' in df_filtered.columns:
                st.subheader("Efficacité des canaux par type de donneur")
                
                # Créer un dataframe croisé
                canal_efficacite = pd.crosstab(
                    df_filtered['Comment_informé'],
                    df_filtered['experience_don'],
                    normalize='index'
                ) * 100
                
                # Renommer les colonnes
                canal_efficacite.columns = ['Primo-donneurs (%)', 'Donneurs réguliers (%)']
                canal_efficacite = canal_efficacite.reset_index()
                
                # Ajouter le nombre total pour chaque canal
                canal_total = df_filtered['Comment_informé'].value_counts().reset_index()
                canal_total.columns = ['Comment_informé', 'Total']
                canal_efficacite = canal_efficacite.merge(canal_total, on='Comment_informé')
                
                # Trier par nombre total décroissant
                canal_efficacite = canal_efficacite.sort_values('Total', ascending=False)
                
                # Graphique de comparaison
                fig_comp = px.bar(
                    canal_efficacite,
                    x='Comment_informé',
                    y=['Primo-donneurs (%)', 'Donneurs réguliers (%)'],
                    title="Efficacité des canaux pour attirer de nouveaux donneurs vs fidéliser",
                    barmode='group',
                    color_discrete_sequence=['#3498db', '#e74c3c'],
                    hover_data=['Total']
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Recommandations basées sur les données
                st.subheader("Recommandations pour la communication")
                
                # Identifier les canaux les plus efficaces pour les nouveaux donneurs
                top_primo = canal_efficacite.sort_values('Primo-donneurs (%)', ascending=False).head(3)
                
                # Identifier les canaux les plus efficaces pour les donneurs réguliers
                top_regulier = canal_efficacite.sort_values('Donneurs réguliers (%)', ascending=False).head(3)
                
                # Afficher les recommandations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Canaux recommandés pour le recrutement**")
                    for i, row in top_primo.iterrows():
                        st.write(f"- {row['Comment_informé']} ({row['Primo-donneurs (%)']:.1f}% de nouveaux donneurs)")
                
                with col2:
                    st.write("**Canaux recommandés pour la fidélisation**")
                    for i, row in top_regulier.iterrows():
                        st.write(f"- {row['Comment_informé']} ({row['Donneurs réguliers (%)']:.1f}% de donneurs réguliers)")
        else:
            st.info("Données insuffisantes sur les canaux de communication.")
        
        # Analyser l'influence de l'entourage
        if 'Influence_entourage' in df_filtered.columns:
            st.subheader("Influence de l'entourage")
            
            # Compter les occurrences
            influence_counts = df_filtered['Influence_entourage'].value_counts().reset_index()
            influence_counts.columns = ['Type d\'influence', 'Nombre de donneurs']
            
            # Graphique
            fig_influence = px.pie(
                influence_counts,
                values='Nombre de donneurs',
                names='Type d\'influence',
                title="Influence de l'entourage sur la décision de donner",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_influence, use_container_width=True)
            
            # Analyse des dons accompagnés
            if 'Accompagné' in df_filtered.columns and 'Nombre_accompagnants' in df_filtered.columns:
                st.subheader("Analyse des dons accompagnés")
                
                # Calculer le pourcentage de dons accompagnés
                accompagne_count = df_filtered[df_filtered['Accompagné'] == 'Oui'].shape[0]
                total_count = df_filtered.shape[0]
                pct_accompagne = (accompagne_count / total_count) * 100 if total_count > 0 else 0
                
                # Calculer le nombre moyen d'accompagnants
                df_filtered['Nombre_accomp_num'] = pd.to_numeric(df_filtered['Nombre_accompagnants'], errors='coerce').fillna(0)
                moyenne_accomp = df_filtered[df_filtered['Accompagné'] == 'Oui']['Nombre_accomp_num'].mean()
                
                # Afficher les métriques
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Pourcentage de dons accompagnés", 
                        f"{pct_accompagne:.1f}%",
                        help="Pourcentage de donneurs qui sont venus accompagnés"
                    )
                
                with col2:
                    st.metric(
                        "Nombre moyen d'accompagnants", 
                        f"{moyenne_accomp:.1f}",
                        help="Nombre moyen de personnes accompagnant chaque donneur"
                    )
                
                # Analyser si les accompagnants donnent aussi
                if 'Don_accompagnants' in df_filtered.columns:
                    don_accomp_counts = df_filtered['Don_accompagnants'].value_counts().reset_index()
                    don_accomp_counts.columns = ['Don des accompagnants', 'Nombre']
                    
                    # Graphique
                    fig_don_accomp = px.pie(
                        don_accomp_counts,
                        values='Nombre',
                        names='Don des accompagnants',
                        title="Les accompagnants ont-ils également donné leur sang?",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_don_accomp, use_container_width=True)
    
    # Tab 3: Motivations des donneurs
    with tab3:
        st.subheader("Analyse des motivations des donneurs")
        
        # Analyser les motivations principales
        if 'Motivation_principale' in df_filtered.columns:
            # Compter les occurrences de chaque motivation
            motiv_counts = df_filtered['Motivation_principale'].value_counts().reset_index()
            motiv_counts.columns = ['Motivation', 'Nombre de donneurs']
            
            # Trier pour avoir les plus fréquentes en premier
            motiv_counts = motiv_counts.sort_values('Nombre de donneurs', ascending=False)
            
            # Graphique des motivations
            fig_motiv = px.bar(
                motiv_counts,
                x='Motivation',
                y='Nombre de donneurs',
                title="Motivations principales des donneurs",
                color='Nombre de donneurs',
                color_continuous_scale='Viridis',
                text_auto=True
            )
            fig_motiv.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_motiv, use_container_width=True)
            
            # Comparer les motivations entre primo-donneurs et donneurs réguliers
            if 'experience_don' in df_filtered.columns:
                st.subheader("Motivations par type de donneur")
                
                # Créer un dataframe croisé
                motiv_type = pd.crosstab(
                    df_filtered['Motivation_principale'],
                    df_filtered['experience_don']
                )
                
                # Renommer les colonnes
                motiv_type.columns = ['Primo-donneurs', 'Donneurs réguliers']
                motiv_type = motiv_type.reset_index()
                
                # Calculer les pourcentages
                primo_total = motiv_type['Primo-donneurs'].sum()
                reg_total = motiv_type['Donneurs réguliers'].sum()
                
                motiv_type['Primo %'] = (motiv_type['Primo-donneurs'] / primo_total * 100).round(1)
                motiv_type['Réguliers %'] = (motiv_type['Donneurs réguliers'] / reg_total * 100).round(1)
                
                # Trier par motivation la plus fréquente
                motiv_type['Total'] = motiv_type['Primo-donneurs'] + motiv_type['Donneurs réguliers']
                motiv_type = motiv_type.sort_values('Total', ascending=False)
                
                # Graphique de comparaison
                fig_comp_motiv = px.bar(
                    motiv_type,
                    x='Motivation_principale',
                    y=['Primo %', 'Réguliers %'],
                    title="Comparaison des motivations entre primo-donneurs et donneurs réguliers",
                    barmode='group',
                    color_discrete_sequence=['#3498db', '#e74c3c']
                )
                fig_comp_motiv.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_comp_motiv, use_container_width=True)
                
                # Tableau des différences significatives
                st.subheader("Différences de motivation significatives")
                
                # Calculer les différences
                motiv_type['Différence'] = motiv_type['Réguliers %'] - motiv_type['Primo %']
                motiv_type_diff = motiv_type[['Motivation_principale', 'Primo %', 'Réguliers %', 'Différence']]
                motiv_type_diff = motiv_type_diff.sort_values('Différence', ascending=False)
                
                # Afficher le tableau
                st.dataframe(motiv_type_diff)
            
            # Analyser la relation avec l'appartenance à un groupe
            if 'Appartenance_groupe' in df_filtered.columns:
                st.subheader("Influence de l'appartenance à un groupe")
                
                # Compter les occurrences
                groupe_counts = df_filtered['Appartenance_groupe'].value_counts().reset_index()
                groupe_counts.columns = ['Appartenance à un groupe', 'Nombre de donneurs']
                
                # Graphique
                fig_groupe = px.pie(
                    groupe_counts,
                    values='Nombre de donneurs',
                    names='Appartenance à un groupe',
                    title="Appartenance à un groupe ou une association",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_groupe, use_container_width=True)
                
                # Analyser l'impact sur la motivation
                motiv_groupe = pd.crosstab(
                    df_filtered['Motivation_principale'],
                    df_filtered['Appartenance_groupe'],
                    normalize='columns'
                ) * 100
                motiv_groupe = motiv_groupe.reset_index()
                
                # Sélectionner les principales motivations
                top_motivations = motiv_counts.head(5)['Motivation'].tolist()
                motiv_groupe_filtered = motiv_groupe[motiv_groupe['Motivation_principale'].isin(top_motivations)]
                
                # Restructurer pour Plotly
                melted_df = pd.melt(
                    motiv_groupe_filtered, 
                    id_vars=['Motivation_principale'], 
                    var_name='Appartenance', 
                    value_name='Pourcentage'
                )
                
                # Graphique
                fig_motiv_groupe = px.bar(
                    melted_df,
                    x='Motivation_principale',
                    y='Pourcentage',
                    color='Appartenance',
                    title="Influence de l'appartenance à un groupe sur les motivations",
                    barmode='group'
                )
                fig_motiv_groupe.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_motiv_groupe, use_container_width=True)
        else:
            st.info("Données insuffisantes sur les motivations des donneurs.")
        
        # Analyser l'intention de don futur
        if 'Intention_don_futur' in df_filtered.columns:
            st.subheader("Intention de don futur")
            
            # Compter les occurrences
            intention_counts = df_filtered['Intention_don_futur'].value_counts().reset_index()
            intention_counts.columns = ['Intention', 'Nombre de donneurs']
            
            # Graphique
            fig_intention = px.pie(
                intention_counts,
                values='Nombre de donneurs',
                names='Intention',
                title="Intention de donner à nouveau dans le futur",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_intention, use_container_width=True)
            
            # Analyser la volonté de recommander
            if 'Prêt_recommander' in df_filtered.columns:
                # Compter les occurrences
                recomm_counts = df_filtered['Prêt_recommander'].value_counts().reset_index()
                recomm_counts.columns = ['Recommandation', 'Nombre de donneurs']
                
                # Graphique
                fig_recomm = px.pie(
                    recomm_counts,
                    values='Nombre de donneurs',
                    names='Recommandation',
                    title="Volonté de recommander le don de sang à l'entourage",
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                st.plotly_chart(fig_recomm, use_container_width=True)
    
    # Tab 4: Analyse des sentiments
    with tab4:
        st.subheader("Analyse des sentiments des donneurs")
        
        # Vérifier s'il y a des colonnes de satisfaction
        satisfaction_cols = [
            'Satisfaction_don_précédent',
            'Satisfaction_globale',
            'Satisfaction_personnel',
            'Confort_installation',
            'Facilité_accès',
            'Temps_attente_perçu'
        ]
        
        available_cols = [col for col in satisfaction_cols if col in df_filtered.columns]
        
        if available_cols:
            # Convertir les colonnes en numérique
            for col in available_cols:
                try:
                    df_filtered[col + '_num'] = pd.to_numeric(df_filtered[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Impossible de convertir {col} en numérique: {e}")
            
            # Afficher les scores moyens de satisfaction
            st.subheader("Scores moyens de satisfaction")
            
            # Créer un dataframe pour les scores moyens
            satisfaction_means = {}
            
            for col in available_cols:
                col_num = col + '_num'
                if col_num in df_filtered.columns:
                    mean_val = df_filtered[col_num].mean()
                    if not pd.isna(mean_val):
                        # Obtenir un nom plus lisible pour l'affichage
                        display_name = col.replace('_', ' ').title()
                        satisfaction_means[display_name] = mean_val
            
            if satisfaction_means:
                # Convertir en dataframe
                satisfaction_df = pd.DataFrame({
                    'Aspect': list(satisfaction_means.keys()),
                    'Score moyen': list(satisfaction_means.values())
                })
                
                # Trier par score
                satisfaction_df = satisfaction_df.sort_values('Score moyen', ascending=False)
                
                # Graphique
                fig_satis = px.bar(
                    satisfaction_df,
                    x='Aspect',
                    y='Score moyen',
                    title="Scores moyens de satisfaction (sur 5)",
                    color='Score moyen',
                    color_continuous_scale='RdYlGn',
                    text_auto='.2f'
                )
                fig_satis.update_traces(textposition='outside')
                fig_satis.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_satis, use_container_width=True)
            
            # Analyser les problèmes rencontrés
            if 'Problèmes_don_précédent' in df_filtered.columns:
                st.subheader("Problèmes rencontrés lors de dons précédents")
                
                # Compter les occurrences
                problemes_counts = df_filtered['Problèmes_don_précédent'].value_counts().reset_index()
                problemes_counts.columns = ['Problèmes rencontrés', 'Nombre de donneurs']
                
                # Graphique
                fig_problemes = px.pie(
                    problemes_counts,
                    values='Nombre de donneurs',
                    names='Problèmes rencontrés',
                    title="Problèmes rencontrés lors de dons précédents",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_problemes, use_container_width=True)
                
                # Analyser les types de problèmes
                if 'Type_problèmes_précédents' in df_filtered.columns:
                    df_avec_problemes = df_filtered[df_filtered['Problèmes_don_précédent'] == 'Oui']
                    
                    if len(df_avec_problemes) > 0:
                        # Compter les types de problèmes
                        types_problemes_counts = df_avec_problemes['Type_problèmes_précédents'].value_counts().reset_index()
                        types_problemes_counts.columns = ['Type de problème', 'Fréquence']
                        
                        # Graphique
                        fig_types = px.bar(
                            types_problemes_counts,
                            x='Type de problème',
                            y='Fréquence',
                            title="Types de problèmes rencontrés",
                            color='Fréquence',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_types, use_container_width=True)
            
            # Analyse des commentaires en texte libre (si disponibles)
            comment_cols = [col for col in df_filtered.columns if 'commentaire' in col.lower() or 'comment' in col.lower()]
            
            if comment_cols:
                st.subheader("Analyse des commentaires en texte libre")
                
                # Fonction pour extraire les mots-clés des commentaires
                def extract_keywords(text_series):
                    # Concaténer tous les commentaires
                    all_text = ' '.join(text_series.dropna().astype(str))
                    
                    # Tokenizer le texte
                    tokens = word_tokenize(all_text.lower())
                    
                    # Filtrer les stop words et la ponctuation
                    stop_words = set(stopwords.words('french') + stopwords.words('english'))
                    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
                    
                    # Compter les occurrences
                    word_counts = Counter(filtered_tokens)
                    
                    return word_counts
                
                # Analyse de sentiment
                def analyze_sentiment(text_series):
                    # Initialiser l'analyseur de sentiment
                    sia = SentimentIntensityAnalyzer()
                    
                    # Calculer le sentiment pour chaque commentaire
                    sentiments = []
                    for comment in text_series.dropna():
                        try:
                            score = sia.polarity_scores(str(comment))
                            sentiments.append(score)
                        except:
                            continue
                    
                    # Calculer les moyennes
                    if sentiments:
                        avg_sentiment = {
                            'pos': sum(s['pos'] for s in sentiments) / len(sentiments),
                            'neu': sum(s['neu'] for s in sentiments) / len(sentiments),
                            'neg': sum(s['neg'] for s in sentiments) / len(sentiments),
                            'compound': sum(s['compound'] for s in sentiments) / len(sentiments)
                        }
                        return avg_sentiment, len(sentiments)
                    
                    return None, 0
                
                # Appliquer l'analyse à chaque colonne de commentaires
                for col in comment_cols:
                    if col in df_filtered.columns:
                        st.write(f"**Analyse des {col}**")
                        
                        # Compter le nombre de commentaires non vides
                        non_empty = df_filtered[col].dropna().shape[0]
                        total = df_filtered.shape[0]
                        pct_with_comments = (non_empty / total) * 100 if total > 0 else 0
                        
                        st.write(f"{non_empty} commentaires sur {total} donneurs ({pct_with_comments:.1f}%)")
                        
                        if non_empty > 0:
                            # Extraire et afficher les mots-clés
                            keywords = extract_keywords(df_filtered[col])
                            
                            # Afficher les 15 mots les plus fréquents
                            top_words = pd.DataFrame({
                                'Mot': [word for word, count in keywords.most_common(15)],
                                'Fréquence': [count for word, count in keywords.most_common(15)]
                            })
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Graphique des mots-clés
                                fig_words = px.bar(
                                    top_words,
                                    x='Mot',
                                    y='Fréquence',
                                    title="Mots-clés les plus fréquents",
                                    color='Fréquence',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig_words, use_container_width=True)
                            
                            with col2:
                                # Analyse de sentiment
                                sentiment, count = analyze_sentiment(df_filtered[col])
                                
                                if sentiment:
                                    st.write("**Analyse de sentiment**")
                                    
                                    # Créer un gauge pour le sentiment global
                                    fig_gauge = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=sentiment['compound'] * 100,
                                        domain={'x': [0, 1], 'y': [0, 1]},
                                        title={'text': "Sentiment global"},
                                        gauge={
                                            'axis': {'range': [-100, 100]},
                                            'bar': {'color': "royalblue"},
                                            'steps': [
                                                {'range': [-100, -50], 'color': 'firebrick'},
                                                {'range': [-50, 0], 'color': 'lightcoral'},
                                                {'range': [0, 50], 'color': 'lightgreen'},
                                                {'range': [50, 100], 'color': 'forestgreen'}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': sentiment['compound'] * 100
                                            }
                                        }
                                    ))
                                    
                                    fig_gauge.update_layout(height=250)
                                    st.plotly_chart(fig_gauge, use_container_width=True)
                                    
                                    # Afficher les détails du sentiment
                                    sentiment_df = pd.DataFrame({
                                        'Aspect': ['Positif', 'Neutre', 'Négatif'],
                                        'Score': [sentiment['pos'], sentiment['neu'], sentiment['neg']]
                                    })
                                    
                                    fig_sentiment = px.bar(
                                        sentiment_df,
                                        x='Aspect',
                                        y='Score',
                                        color='Aspect',
                                        color_discrete_map={
                                            'Positif': 'green',
                                            'Neutre': 'grey',
                                            'Négatif': 'red'
                                        }
                                    )
                                    st.plotly_chart(fig_sentiment, use_container_width=True)
                            
                            # Afficher un échantillon des commentaires
                            with st.expander("Voir un échantillon de commentaires"):
                                sample_size = min(5, non_empty)
                                sample_comments = df_filtered[col].dropna().sample(sample_size)
                                
                                for i, comment in enumerate(sample_comments):
                                    st.write(f"{i+1}. {comment}")
        else:
            st.info("Données insuffisantes pour l'analyse des sentiments des donneurs.")

if __name__ == "__main__":
    analyse_donneurs()