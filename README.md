# Tableau de bord d'analyse des campagnes de don de sang

Ce projet fournit un tableau de bord complet, implémenté en Python, pour la visualisation et l'analyse des données des campagnes de don de sang.

## Fonctionnalités

1. **Carte de répartition des donneurs**
   - Visualisation géographique des donneurs par arrondissement et quartier
   - Identification des zones à forte ou faible participation

2. **Conditions de santé et éligibilité**
   - Analyse de l'impact des problèmes de santé sur l'admissibilité au don
   - Visualisation des taux d'éligibilité par condition de santé

3. **Profilage des donateurs idéaux**
   - Clustering des donneurs selon leurs caractéristiques démographiques et de santé
   - Identification des profils de donneurs les plus courants

4. **Efficacité des campagnes**
   - Analyse des tendances temporelles des dons
   - Identification des périodes optimales pour les campagnes

5. **Fidélisation des donateurs**
   - Analyse des facteurs influençant les dons répétés
   - Visualisation des caractéristiques des donneurs fidèles

6. **Analyse des sentiments**
   - Analyse des commentaires textuels des donneurs
   - Visualisation des tendances de sentiment

7. **Prédiction d'éligibilité**
   - Modèle d'apprentissage automatique pour prédire l'éligibilité des donneurs
   - API pour des prédictions en temps réel

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone <url-du-repo>
   cd blood_donation_dashboard
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Placez votre fichier de données CSV dans le dossier `data/`.

## Utilisation

1. Lancez l'application Streamlit :
   ```bash
   streamlit run app.py
   ```

2. Pour utiliser l'API de prédiction (optionnel) :
   ```bash
   cd api
   uvicorn api:app --reload
   ```

## Structure du projet

```
blood_donation_dashboard/
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances
├── README.md               # Documentation
├── data/                   # Données
│   └── blood_donation.csv  # Données des campagnes
├── src/                    # Code source
│   ├── data_processing.py  # Traitement des données
│   ├── visualization.py    # Fonctions de visualisation
│   ├── ml_models.py        # Modèles d'apprentissage automatique
│   └── utils.py            # Fonctions utilitaires
├── api/                    # API pour le modèle prédictif
│   └── api.py              # API FastAPI
└── model/                  # Modèles entraînés
    └── eligibility_model.pkl  # Modèle d'éligibilité
```

## Librairies principales utilisées

- **Streamlit** : Interface utilisateur interactive
- **Plotly** : Visualisations interactives
- **Folium** : Cartes géographiques
- **Scikit-learn** : Modèles d'apprentissage automatique
- **FastAPI** : API pour les prédictions
- **Pandas & NumPy** : Traitement des données
- **NLTK & TextBlob** : Analyse des sentiments

## Confidentialité des données

Ce projet veille à ce que toutes les informations personnelles contenues dans l'ensemble de données soient anonymisées et traitées conformément aux directives appropriées.

## Auteur

[Votre nom]

## Licence

Ce projet est sous licence [spécifiez votre licence].
