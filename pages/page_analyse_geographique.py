import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def load_data():
    """Charger les données du fichier CSV enrichi"""
    try:
        df = pd.read_csv('dataset_don_sang_enrichi.csv', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

def map():
    st.title("Analyse Géographique des Dons de Sang")
    
    # Chargement des données
    df = load_data()
    
    if df is None:
        st.warning("Impossible de charger les données. Veuillez vérifier que le fichier 'dataset_don_sang_enrichi.csv' existe.")
        return
    
    # Sidebar pour les filtres
    st.sidebar.title("Filtres")
    
    # Filtre de période
    st.sidebar.subheader("Période")
    # Déterminer les dates min et max dans les données
    if 'Date_don' in df.columns:
        df['Date_don'] = pd.to_datetime(df['Date_don'], errors='coerce')
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
    
    # Filtre par campagne
    if 'ID_campagne' in df.columns:
        campagnes = ['Toutes'] + sorted(df['ID_campagne'].unique().tolist())
        campagne_selectionnee = st.sidebar.selectbox("Campagne", campagnes)
    
    # Filtre par groupe sanguin
    if 'Groupe_sanguin' in df.columns:
        groupes_sanguins = ['Tous'] + sorted(df['Groupe_sanguin'].unique().tolist())
        groupe_sanguin_selectionne = st.sidebar.selectbox("Groupe sanguin", groupes_sanguins)
    
    # Filtre par statut de fidélité
    statut_donneur = st.sidebar.radio(
        "Statut du donneur",
        ['Tous', 'Primo-donneurs', 'Donneurs réguliers']
    )
    
    # Filtrage des données
    df_filtered = df.copy()
    
    # Filtrage par date
    if 'Date_don' in df.columns:
        df_filtered = df_filtered[
            (df_filtered['Date_don'] >= pd.Timestamp(date_debut)) & 
            (df_filtered['Date_don'] <= pd.Timestamp(date_fin))
        ]
    
    # Filtrage par campagne
    if 'ID_campagne' in df.columns and campagne_selectionnee != 'Toutes':
        df_filtered = df_filtered[df_filtered['ID_campagne'] == campagne_selectionnee]
    
    # Filtrage par groupe sanguin
    if 'Groupe_sanguin' in df.columns and groupe_sanguin_selectionne != 'Tous':
        df_filtered = df_filtered[df_filtered['Groupe_sanguin'] == groupe_sanguin_selectionne]
    
    # Filtrage par statut de fidélité
    if 'experience_don' in df.columns and statut_donneur != 'Tous':
        if statut_donneur == 'Primo-donneurs':
            df_filtered = df_filtered[df_filtered['experience_don'] == 0]
        elif statut_donneur == 'Donneurs réguliers':
            df_filtered = df_filtered[df_filtered['experience_don'] == 1]
    
    # Vérifier si les données filtrées sont vides
    if len(df_filtered) == 0:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Onglets pour naviguer entre les différentes analyses géographiques
    tab1, tab2, tab3, tab4 = st.tabs([
        "Répartition des donneurs", 
        "Fidélité par zone", 
        "Zones à cibler", 
        "Groupes sanguins par zone"
    ])
    
    # Créer un dictionnaire de coordonnées géographiques pour les arrondissements de Douala
    # Ces coordonnées sont approximatives et devraient être remplacées par des données réelles
    coords = {
        'Douala 1': (4.0581, 9.7133),
        'Douala 2': (4.0711, 9.7222),
        'Douala 3': (4.0535, 9.7521),
        'Douala 4': (4.0736, 9.6903),
        'Douala 5': (4.0229, 9.7566),
        'Douala (Non précisé)': (4.0511, 9.7679)
    }
    
    # Tab 1: Répartition des donneurs par arrondissement
    with tab1:
        st.subheader("Répartition des donneurs par zone géographique")
        
        # Obtenir la répartition par arrondissement
        if 'arrondissement_clean' in df_filtered.columns:
            repartition_arr = df_filtered['arrondissement_clean'].value_counts().reset_index()
            repartition_arr.columns = ['Arrondissement', 'Nombre de donneurs']
            
            # Graphique à barres des donneurs par arrondissement
            fig_arr = px.bar(
                repartition_arr,
                x='Arrondissement',
                y='Nombre de donneurs',
                color='Nombre de donneurs',
                title="Nombre de donneurs par arrondissement",
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_arr, use_container_width=True)
            
            # Carte de densité des donneurs avec Plotly
            st.subheader("Carte de densité des donneurs")
            
            # Ajouter des coordonnées à notre dataframe
            repartition_arr['lat'] = repartition_arr['Arrondissement'].map(lambda x: coords.get(x, (4.0511, 9.7679))[0])
            repartition_arr['lon'] = repartition_arr['Arrondissement'].map(lambda x: coords.get(x, (4.0511, 9.7679))[1])
            
            # Normaliser la taille des marqueurs pour une meilleure visualisation
            max_donneurs = repartition_arr['Nombre de donneurs'].max()
            repartition_arr['Taille_marqueur'] = repartition_arr['Nombre de donneurs'] / max_donneurs * 50 + 10
            
            # Créer la carte avec Plotly
            fig_map = px.scatter_mapbox(
                repartition_arr,
                lat="lat",
                lon="lon",
                size="Taille_marqueur",
                color="Nombre de donneurs",
                hover_name="Arrondissement",
                hover_data=["Nombre de donneurs"],
                color_continuous_scale="Reds",
                size_max=50,
                zoom=11.5,
                center={"lat": 4.0511, "lon": 9.7679},
                mapbox_style="open-street-map",
                title="Densité des donneurs par arrondissement"
            )
            
            # Ajouter des annotations pour chaque arrondissement
            annotations = []
            for i, row in repartition_arr.iterrows():
                annotations.append(
                    dict(
                        x=row['lon'],
                        y=row['lat'],
                        text=row['Arrondissement'],
                        showarrow=False,
                        font=dict(color="black", size=10),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=2,
                        opacity=0.8
                    )
                )
            
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                annotations=annotations
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Répartition par quartier (top 10)
            if 'quartier_clean' in df_filtered.columns:
                st.subheader("Top 10 des quartiers par nombre de donneurs")
                repartition_quartier = df_filtered['quartier_clean'].value_counts().nlargest(10).reset_index()
                repartition_quartier.columns = ['Quartier', 'Nombre de donneurs']
                
                fig_quartier = px.bar(
                    repartition_quartier,
                    x='Quartier',
                    y='Nombre de donneurs',
                    color='Nombre de donneurs',
                    title="Top 10 des quartiers par nombre de donneurs",
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_quartier, use_container_width=True)
    
    # Tab 2: Fidélité par zone géographique
    with tab2:
        st.subheader("Fidélité des donneurs par zone géographique")
        
        if 'arrondissement_clean' in df_filtered.columns and 'experience_don' in df_filtered.columns:
            # Calculer le taux de fidélité par arrondissement
            fidelite_arr = df_filtered.groupby('arrondissement_clean')['experience_don'].mean().reset_index()
            fidelite_arr.columns = ['Arrondissement', 'Taux de fidélité']
            fidelite_arr['Taux de fidélité'] = fidelite_arr['Taux de fidélité'] * 100  # Convertir en pourcentage
            
            # Graphique des taux de fidélité par arrondissement
            fig_fidelite = px.bar(
                fidelite_arr.sort_values('Taux de fidélité', ascending=False),
                x='Arrondissement',
                y='Taux de fidélité',
                color='Taux de fidélité',
                title="Taux de fidélité par arrondissement (%)",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_fidelite, use_container_width=True)
            
            # Carte des zones avec les donneurs les plus fidèles
            st.subheader("Carte des zones avec donneurs les plus fidèles")
            
            # Ajouter des coordonnées à notre dataframe
            fidelite_arr['lat'] = fidelite_arr['Arrondissement'].map(lambda x: coords.get(x, (4.0511, 9.7679))[0])
            fidelite_arr['lon'] = fidelite_arr['Arrondissement'].map(lambda x: coords.get(x, (4.0511, 9.7679))[1])
            
            # Créer la carte avec Plotly
            fig_map_fidelite = px.scatter_mapbox(
                fidelite_arr,
                lat="lat",
                lon="lon",
                size="Taux de fidélité",
                color="Taux de fidélité",
                hover_name="Arrondissement",
                hover_data=["Taux de fidélité"],
                color_continuous_scale="Blues",
                size_max=40,
                zoom=11.5,
                center={"lat": 4.0511, "lon": 9.7679},
                mapbox_style="open-street-map",
                title="Taux de fidélité par arrondissement"
            )
            
            # Ajouter des annotations pour chaque arrondissement
            annotations = []
            for i, row in fidelite_arr.iterrows():
                annotations.append(
                    dict(
                        x=row['lon'],
                        y=row['lat'],
                        text=f"{row['Arrondissement']}: {row['Taux de fidélité']:.1f}%",
                        showarrow=False,
                        font=dict(color="black", size=10),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=2,
                        opacity=0.8
                    )
                )
            
            fig_map_fidelite.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                annotations=annotations
            )
            
            st.plotly_chart(fig_map_fidelite, use_container_width=True)
            
            # Tableau des arrondissements avec le plus haut taux de fidélité
            st.subheader("Classement des arrondissements par fidélité")
            st.dataframe(fidelite_arr.sort_values('Taux de fidélité', ascending=False))
    
    # Tab 3: Zones nécessitant plus d'efforts de recrutement
    with tab3:
        st.subheader("Zones nécessitant plus d'efforts de recrutement")
        
        if 'arrondissement_clean' in df_filtered.columns:
            # Population estimée par arrondissement (données fictives pour l'exemple)
            # En situation réelle, utilisez des données démographiques réelles
            population_arr = {
                'Douala 1': 250000,
                'Douala 2': 350000,
                'Douala 3': 650000,
                'Douala 4': 300000,
                'Douala 5': 550000,
                'Douala (Non précisé)': 100000
            }
            
            # Calculer le taux de pénétration par arrondissement
            repartition_arr = df_filtered['arrondissement_clean'].value_counts().reset_index()
            repartition_arr.columns = ['Arrondissement', 'Nombre de donneurs']
            repartition_arr['Population estimée'] = repartition_arr['Arrondissement'].map(
                lambda x: population_arr.get(x, 100000)
            )
            repartition_arr['Taux de pénétration (‰)'] = (repartition_arr['Nombre de donneurs'] / repartition_arr['Population estimée']) * 1000
            
            # Graphique du taux de pénétration par arrondissement
            fig_penetration = px.bar(
                repartition_arr.sort_values('Taux de pénétration (‰)'),
                x='Arrondissement',
                y='Taux de pénétration (‰)',
                color='Taux de pénétration (‰)',
                title="Taux de pénétration des dons par arrondissement (‰)",
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig_penetration, use_container_width=True)
            
            # Calculer la priorité de recrutement (inversement proportionnelle au taux de pénétration)
            repartition_arr['Priorité de recrutement'] = 10 - repartition_arr['Taux de pénétration (‰)']
            repartition_arr['Priorité de recrutement'] = repartition_arr['Priorité de recrutement'].apply(
                lambda x: max(0, min(10, x))  # Limiter entre 0 et 10
            )
            
            # Carte des zones prioritaires pour le recrutement
            st.subheader("Zones prioritaires pour le recrutement")
            
            # Ajouter des coordonnées à notre dataframe
            repartition_arr['lat'] = repartition_arr['Arrondissement'].map(lambda x: coords.get(x, (4.0511, 9.7679))[0])
            repartition_arr['lon'] = repartition_arr['Arrondissement'].map(lambda x: coords.get(x, (4.0511, 9.7679))[1])
            
            # Créer la carte avec Plotly
            fig_map_priorite = px.scatter_mapbox(
                repartition_arr,
                lat="lat",
                lon="lon",
                size="Priorité de recrutement",
                color="Priorité de recrutement",
                hover_name="Arrondissement",
                hover_data=["Nombre de donneurs", "Population estimée", "Taux de pénétration (‰)", "Priorité de recrutement"],
                color_continuous_scale="RdYlGn_r",  # Rouge = haute priorité, vert = faible priorité
                size_max=40,
                zoom=11.5,
                center={"lat": 4.0511, "lon": 9.7679},
                mapbox_style="open-street-map",
                title="Priorité de recrutement par arrondissement"
            )
            
            # Ajouter des annotations pour chaque arrondissement
            annotations = []
            for i, row in repartition_arr.iterrows():
                annotations.append(
                    dict(
                        x=row['lon'],
                        y=row['lat'],
                        text=f"{row['Arrondissement']}: {row['Priorité de recrutement']:.1f}/10",
                        showarrow=False,
                        font=dict(color="black", size=10),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=2,
                        opacity=0.8
                    )
                )
            
            fig_map_priorite.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                annotations=annotations
            )
            
            st.plotly_chart(fig_map_priorite, use_container_width=True)
            
            # Tableau des zones prioritaires
            st.subheader("Classement des zones prioritaires pour le recrutement")
            zones_prioritaires = repartition_arr[['Arrondissement', 'Nombre de donneurs', 'Population estimée', 'Taux de pénétration (‰)', 'Priorité de recrutement']]
            st.dataframe(zones_prioritaires.sort_values('Priorité de recrutement', ascending=False))
    
    # Tab 4: Groupes sanguins par zone
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

if __name__ == "__main__":
    map()