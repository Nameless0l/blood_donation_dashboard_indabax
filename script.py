import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re

def enrichir_dataset_don_sang(chemin_fichier_source, chemin_fichier_enrichi):
    """
    Enrichit le dataset existant avec de nouvelles colonnes pour le tableau de bord
    
    Args:
        chemin_fichier_source: Chemin vers le fichier CSV original
        chemin_fichier_enrichi: Chemin où sauvegarder le nouveau fichier CSV
    """
    print("Chargement du dataset original...")
    
    # Charger le dataset original
    try:
        df = pd.read_csv(chemin_fichier_source, encoding='utf-8')
        print(f"Dataset chargé avec succès. Nombre d'entrées: {len(df)}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return
    
    # Créer une copie pour ne pas modifier l'original
    df_enrichi = df.copy()
    
    # Fixons d'abord quelques problèmes potentiels dans les données
    print("Préparation des données existantes...")
    
    # S'assurer que les colonnes de dates sont au format datetime
    for col in ['Date de remplissage de la fiche', 'Date de naissance', 'Si oui preciser la date du dernier don.']:
        if col in df_enrichi.columns:
            df_enrichi[col] = pd.to_datetime(df_enrichi[col], errors='coerce')
    
    # Utiliser la date de remplissage comme date du don si celle-ci est disponible
    # Sinon, utiliser la date actuelle pour simuler
    if 'Date de remplissage de la fiche' in df_enrichi.columns:
        df_enrichi['Date_don'] = df_enrichi['Date de remplissage de la fiche']
    else:
        # Si pas de date de remplissage, créer une date simulée récente
        df_enrichi['Date_don'] = datetime.now() - pd.to_timedelta(np.random.randint(0, 60, size=len(df_enrichi)), unit='d')
    
    # Vérifier si le champ 'experience_don' est présent, sinon le créer
    if 'experience_don' not in df_enrichi.columns:
        df_enrichi['experience_don'] = np.where(df_enrichi['A-t-il (elle) déjà donné le sang'] == 'Oui', 1, 0)
    
    print("Ajout des nouvelles colonnes...")
    
    # 1. Identification et contact
    # ---------------------------
    # Générer des numéros de téléphone fictifs (format Cameroun)
    df_enrichi['Numéro_téléphone'] = [f"6{random.randint(5, 9)}{random.randint(1000000, 9999999)}" for _ in range(len(df_enrichi))]
    df_enrichi['Consentement_contact'] = np.random.choice(['Oui', 'Non'], size=len(df_enrichi), p=[0.9, 0.1])
    
    # 2. Données sur le don actuel
    # ---------------------------
    # Créer un ID de campagne (format: CAMP_ANNEE_MOIS_LIEU)
    dates_don = pd.to_datetime(df_enrichi['Date_don']).fillna(datetime.now())
    
    # Sélectionner des lieux de campagne fictifs basés sur l'arrondissement
    lieux_campagne = {
        'Douala 1': ['Hôpital Laquintinie', 'Place du Gouvernement', 'Lycée Joss'],
        'Douala 2': ['Hôpital Général', 'New Bell', 'Marché Central'],
        'Douala 3': ['Centre de santé Logbaba', 'Université', 'Église St Pierre'],
        'Douala 4': ['Centre médical Bonassama', 'Marché Bonabéri', 'Lycée Bonabéri'],
        'Douala 5': ['Hôpital de district Logbessou', 'Carrefour Ndokoti', 'Centre commercial'],
        'default': ['Centre de santé mobile', 'Université', 'Place publique', 'Centre commercial']
    }
    
    # Fonction pour obtenir un lieu de campagne selon l'arrondissement
    def get_lieu_campagne(arr):
        if isinstance(arr, str) and any(d in arr for d in ['1', '2', '3', '4', '5']):
            for key in lieux_campagne.keys():
                if key != 'default' and key in arr:
                    return random.choice(lieux_campagne[key])
        return random.choice(lieux_campagne['default'])
    
    # Créer IDs de campagne et lieux
    campagnes = []
    lieux_don = []
    
    for i, row in df_enrichi.iterrows():
        date = dates_don[i] if not pd.isna(dates_don[i]) else datetime.now()
        arr = row['arrondissement_clean'] if 'arrondissement_clean' in df_enrichi.columns and not pd.isna(row['arrondissement_clean']) else 'Douala'
        lieu = get_lieu_campagne(arr)
        lieux_don.append(lieu)
        
        # Format campagne: CAMP_ANNEE_MOIS_CODE
        campagne_id = f"CAMP_{date.year}_{date.month:02d}_{re.sub('[^A-Za-z0-9]', '', lieu.split()[0])}"
        campagnes.append(campagne_id)
    
    df_enrichi['ID_campagne'] = campagnes
    df_enrichi['Lieu_don'] = lieux_don
    
    # Type de don (majoritairement sang total pour les campagnes)
    df_enrichi['Type_don'] = np.random.choice(['Sang total', 'Plasma', 'Plaquettes'], 
                                            size=len(df_enrichi), 
                                            p=[0.9, 0.07, 0.03])
    
    # Volume de don basé sur le genre (en moyenne, hommes: 450ml, femmes: 400ml)
    volumes = []
    for genre in df_enrichi['Genre']:
        if genre == 'Homme':
            volume = random.choice([450, 470, 430, 450, 450])
        else:  # Femme
            volume = random.choice([400, 420, 380, 400, 400])
        volumes.append(volume)
    
    df_enrichi['Volume_don'] = volumes
    
    # Groupe sanguin (distribution réaliste pour l'Afrique)
    df_enrichi['Groupe_sanguin'] = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], 
                                                  size=len(df_enrichi), 
                                                  p=[0.24, 0.03, 0.20, 0.02, 0.04, 0.01, 0.43, 0.03])
    
    # Heures d'arrivée et de départ
    heures_arrivee = []
    heures_depart = []
    
    for _ in range(len(df_enrichi)):
        # Heure d'arrivée entre 8h et 16h
        heure_arrivee = datetime.now().replace(
            hour=random.randint(8, 16),
            minute=random.randint(0, 59),
            second=0
        )
        
        # Durée du processus entre 30 et 120 minutes
        duree = timedelta(minutes=random.randint(30, 120))
        heure_depart = heure_arrivee + duree
        
        heures_arrivee.append(heure_arrivee.strftime('%H:%M'))
        heures_depart.append(heure_depart.strftime('%H:%M'))
    
    df_enrichi['Heure_arrivée'] = heures_arrivee
    df_enrichi['Heure_départ'] = heures_depart
    
    # 3. Expérience précédente (pour donneurs non-premiers)
    # ---------------------------------------------------
    donneurs_non_premiers = df_enrichi['experience_don'] == 1
    
    # Satisfaction du don précédent
    df_enrichi['Satisfaction_don_précédent'] = np.nan
    df_enrichi.loc[donneurs_non_premiers, 'Satisfaction_don_précédent'] = np.random.randint(1, 6, size=donneurs_non_premiers.sum())
    
    # Problèmes rencontrés lors du don précédent
    df_enrichi['Problèmes_don_précédent'] = np.nan
    df_enrichi.loc[donneurs_non_premiers, 'Problèmes_don_précédent'] = np.random.choice([0, 1], 
                                                                                    size=donneurs_non_premiers.sum(), 
                                                                                    p=[0.8, 0.2])
    
    # Types de problèmes
    problemes = ['Malaise', 'Hématome', 'Attente longue', 'Personnel peu aimable', 'Douleur au bras', 'Autre']
    
    df_enrichi['Type_problèmes_précédents'] = np.nan
    mask_problemes = (donneurs_non_premiers) & (df_enrichi['Problèmes_don_précédent'] == 1)
    df_enrichi.loc[mask_problemes, 'Type_problèmes_précédents'] = np.random.choice(problemes, size=mask_problemes.sum())
    
    # 4. Source et motivation
    # ----------------------
    canaux = ['SMS', 'Radio', 'Réseaux sociaux', 'Affiche', 'Bouche-à-oreille', 'TV', 'Email']
    df_enrichi['Comment_informé'] = np.random.choice(canaux, size=len(df_enrichi))
    df_enrichi['Canal_information'] = df_enrichi['Comment_informé']  # Redondant mais conservé pour cohérence
    
    motivations = ['Altruisme', 'Besoin spécifique', 'Connaissance malade', 'Avantages santé', 'Habitude', 'Événement solidaire']
    df_enrichi['Motivation_principale'] = np.random.choice(motivations, size=len(df_enrichi))
    df_enrichi['Influence_entourage'] = np.random.choice(['Oui', 'Non'], size=len(df_enrichi), p=[0.4, 0.6])
    
    # 5. Logistique et accessibilité
    # -----------------------------
    # Distance parcourue (basée sur l'arrondissement)
    distances = []
    for arr in df_enrichi['arrondissement_clean']:
        if isinstance(arr, str):
            if "Douala" in arr:
                # Distance plus courte pour les résidents de Douala
                distances.append(random.randint(1, 15))
            else:
                # Distance plus longue pour les résidents hors Douala
                distances.append(random.randint(10, 40))
        else:
            # Valeur par défaut si arrondissement non spécifié
            distances.append(random.randint(5, 20))
    
    df_enrichi['Distance_parcourue'] = distances
    
    moyens_transport = ['À pied', 'Transport en commun', 'Véhicule personnel', 'Taxi', 'Moto-taxi']
    df_enrichi['Moyen_transport'] = np.random.choice(moyens_transport, size=len(df_enrichi), 
                                                  p=[0.1, 0.45, 0.2, 0.15, 0.1])
    
    df_enrichi['Facilité_accès'] = np.random.randint(1, 6, size=len(df_enrichi))
    df_enrichi['Temps_disponible'] = np.random.choice(['<30min', '30-60min', '1-2h', '>2h'], size=len(df_enrichi))
    
    # 6. Expérience actuelle et intentions
    # -----------------------------------
    df_enrichi['Temps_attente_perçu'] = np.random.choice(['Court', 'Moyen', 'Long'], size=len(df_enrichi), 
                                                      p=[0.3, 0.5, 0.2])
    
    df_enrichi['Confort_installation'] = np.random.randint(1, 6, size=len(df_enrichi))
    df_enrichi['Satisfaction_personnel'] = np.random.randint(1, 6, size=len(df_enrichi))
    df_enrichi['Satisfaction_globale'] = np.random.randint(1, 6, size=len(df_enrichi))
    
    # Intention liée à l'éligibilité
    intentions = []
    for eligible in df_enrichi['ÉLIGIBILITÉ AU DON.']:
        if eligible == 'Eligible':
            intentions.append(np.random.choice(['Oui', 'Peut-être', 'Non'], p=[0.7, 0.25, 0.05]))
        elif eligible == 'Temporairement Non-eligible':
            intentions.append(np.random.choice(['Oui', 'Peut-être', 'Non'], p=[0.5, 0.4, 0.1]))
        else:  # Définitivement non-eligible
            intentions.append(np.random.choice(['Non', 'Peut-être'], p=[0.9, 0.1]))
    
    df_enrichi['Intention_don_futur'] = intentions
    
    df_enrichi['Fréquence_don_souhaitée'] = np.random.choice(['1 fois/an', '2 fois/an', '3-4 fois/an', 'Maximum possible'], 
                                                          size=len(df_enrichi), 
                                                          p=[0.3, 0.4, 0.2, 0.1])
    
    # 7. Aspect communautaire
    # ----------------------
    df_enrichi['Accompagné'] = np.random.choice(['Oui', 'Non'], size=len(df_enrichi), p=[0.3, 0.7])
    
    # Nombre d'accompagnants (si accompagné)
    accompagnants = []
    for acc in df_enrichi['Accompagné']:
        if acc == 'Oui':
            accompagnants.append(random.randint(1, 5))
        else:
            accompagnants.append(0)
    
    df_enrichi['Nombre_accompagnants'] = accompagnants
    
    # Don des accompagnants
    don_acc = []
    for acc, nb_acc in zip(df_enrichi['Accompagné'], df_enrichi['Nombre_accompagnants']):
        if acc == 'Oui':
            if nb_acc == 1:
                don_acc.append(np.random.choice(['Oui', 'Non'], p=[0.6, 0.4]))
            else:
                don_acc.append(np.random.choice(['Oui', 'Non', 'Certains'], p=[0.3, 0.3, 0.4]))
        else:
            don_acc.append(None)
    
    df_enrichi['Don_accompagnants'] = don_acc
    
    # Score NPS (Net Promoter Score) - Prêt à recommander
    df_enrichi['Prêt_recommander'] = np.random.randint(0, 11, size=len(df_enrichi))
    
    # Appartenance à un groupe
    groupes = ['Aucun', 'Entreprise', 'École/Université', 'Association', 'Groupe religieux']
    df_enrichi['Appartenance_groupe'] = np.random.choice(groupes, size=len(df_enrichi), p=[0.6, 0.1, 0.15, 0.05, 0.1])
    
    # 8. Données contextuelles
    # -----------------------
    # Conditions météo (selon les saisons du Cameroun)
    meteo = ['Ensoleillé', 'Nuageux', 'Pluvieux', 'Chaud', 'Humide']
    df_enrichi['Conditions_météo'] = np.random.choice(meteo, size=len(df_enrichi))
    
    # Événements locaux
    evenements = ['Aucun', 'Fête locale', 'Événement sportif', 'Fête nationale', 'Autre']
    df_enrichi['Événement_local'] = np.random.choice(evenements, size=len(df_enrichi), p=[0.7, 0.1, 0.1, 0.05, 0.05])
    
    # Jour de la semaine (basé sur la date du don si disponible)
    jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    jours = []
    
    for date in df_enrichi['Date_don']:
        if pd.notna(date):
            jour_idx = date.weekday() if hasattr(date, 'weekday') else pd.Timestamp(date).weekday()
            jours.append(jours_semaine[jour_idx])
        else:
            jours.append(random.choice(jours_semaine))
    
    df_enrichi['Jour_semaine'] = jours
    
    # Période de la journée (basée sur l'heure d'arrivée)
    periodes = []
    for heure in df_enrichi['Heure_arrivée']:
        if pd.notna(heure):
            heure_val = int(heure.split(':')[0])
            if heure_val < 11:
                periodes.append('Matin')
            elif heure_val < 13:
                periodes.append('Midi')
            elif heure_val < 17:
                periodes.append('Après-midi')
            else:
                periodes.append('Soir')
        else:
            periodes.append(random.choice(['Matin', 'Midi', 'Après-midi', 'Soir']))
    
    df_enrichi['Période_journée'] = periodes
    
    # 9. Suivi potentiel
    # ----------------
    # Date du prochain don possible (3 mois après pour les éligibles)
    dates_prochain = []
    
    for i, row in df_enrichi.iterrows():
        date_don = row['Date_don']
        if pd.notna(date_don):
            if hasattr(date_don, 'date'):
                date_base = date_don.date()
            else:
                date_base = pd.Timestamp(date_don).date()
                
            # Différent délai selon le genre (56 jours hommes, 112 jours femmes)
            if row['Genre'] == 'Homme':
                date_prochain = date_base + timedelta(days=84)  # 3 mois pour les hommes
            else:
                date_prochain = date_base + timedelta(days=112)  # 4 mois pour les femmes
                
            dates_prochain.append(date_prochain)
        else:
            dates_prochain.append(None)
    
    df_enrichi['Date_prochain_don_possible'] = dates_prochain
    
    # Préférence de rappel
    df_enrichi['Préférence_rappel'] = np.random.choice(['SMS', 'Appel', 'Les deux', 'Aucun'], 
                                                    size=len(df_enrichi), 
                                                    p=[0.6, 0.2, 0.15, 0.05])
    
    # Meilleur moment pour contacter
    df_enrichi['Meilleur_moment_contact'] = np.random.choice(['Matin', 'Midi', 'Soir', 'Week-end'], 
                                                          size=len(df_enrichi), 
                                                          p=[0.2, 0.1, 0.5, 0.2])
    
    # Sauvegarder le dataset enrichi
    print(f"Enregistrement du dataset enrichi ({len(df_enrichi)} entrées)...")
    df_enrichi.to_csv(chemin_fichier_enrichi, index=False, encoding='utf-8')
    print(f"Dataset enrichi enregistré avec succès dans {chemin_fichier_enrichi}")
    
    return df_enrichi

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacer ces chemins par les chemins appropriés
    chemin_source = "processed_data/combined_processed.csv"  # Fichier source
    chemin_cible = "dataset_don_sang_enrichi.csv"  # Fichier enrichi à créer
    
    # Exécuter l'enrichissement
    enrichir_dataset_don_sang(chemin_source, chemin_cible)
    
    print("\nColonnes ajoutées au dataset :")
    print("""
1. Identification et contact:
   - Numéro_téléphone
   - Consentement_contact

2. Données sur le don actuel:
   - ID_campagne
   - Date_don
   - Lieu_don
   - Type_don
   - Volume_don
   - Groupe_sanguin
   - Heure_arrivée
   - Heure_départ

3. Expérience précédente (pour donneurs non-premiers):
   - Satisfaction_don_précédent
   - Problèmes_don_précédent
   - Type_problèmes_précédents

4. Source et motivation:
   - Comment_informé
   - Canal_information
   - Motivation_principale
   - Influence_entourage

5. Logistique et accessibilité:
   - Distance_parcourue
   - Moyen_transport
   - Facilité_accès
   - Temps_disponible

6. Expérience actuelle et intentions:
   - Temps_attente_perçu
   - Confort_installation
   - Satisfaction_personnel
   - Satisfaction_globale
   - Intention_don_futur
   - Fréquence_don_souhaitée

7. Aspect communautaire:
   - Accompagné
   - Nombre_accompagnants
   - Don_accompagnants
   - Prêt_recommander
   - Appartenance_groupe

8. Données contextuelles:
   - Conditions_météo
   - Événement_local
   - Jour_semaine
   - Période_journée

9. Suivi potentiel:
   - Date_prochain_don_possible
   - Préférence_rappel
   - Meilleur_moment_contact
    """)

