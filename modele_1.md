# Rapport Technique - Modèle de prédiction d'éligibilité au don de sang

## Table des matières

1. [Introduction](#introduction)
2. [Données utilisées](#données-utilisées)
3. [Prétraitement des données](#prétraitement-des-données)
4. [Exploration et analyse des données](#exploration-et-analyse-des-données)
5. [Développement des modèles](#développement-des-modèles)
6. [Évaluation comparative des modèles](#évaluation-comparative-des-modèles)
7. [Modèle final retenu](#modèle-final-retenu)
8. [Règles de sécurité implémentées](#règles-de-sécurité-implémentées)
9. [Intégration au tableau de bord](#intégration-au-tableau-de-bord)
10. [Limites et perspectives d'amélioration](#limites-et-perspectives-damélioration)

## Introduction

Ce rapport présente le développement et l'évaluation d'un modèle de prédiction d'éligibilité au don de sang pour le projet de tableau de bord de l'IndabaX 2025. L'objectif principal est de créer un modèle fiable capable de prédire si un potentiel donneur est éligible au don de sang en fonction de ses caractéristiques démographiques et médicales, tout en respectant les règles de sécurité transfusionnelle.

## Données utilisées

### Source des données

Les données utilisées pour ce projet proviennent du jeu de données "Updated Challenge dataset.xlsx", fourni dans le cadre du challenge IndabaX 2025. Ce jeu de données contient des informations sur des candidats au don de sang, incluant leurs caractéristiques démographiques, médicales et leur éligibilité au don.

### Structure des données

Le jeu de données comprend trois feuilles principales:
- **Candidat au don 2019 (avec anne)**: Informations sur les candidats
- **Donneurs 2019**: Informations sur les donneurs confirmés
- **Candidat au don 2019 (avec age)**: Version alternative avec âge explicite

La feuille principale utilisée pour l'entraînement du modèle est "Candidat au don 2019 (avec anne)" qui contient **[NOMBRE]** enregistrements et **[NOMBRE]** variables.

### Variables clés

Les principales variables utilisées pour la modélisation sont:

| Variable | Type | Description |
|----------|------|-------------|
| age | Numérique | Âge du candidat en années |
| experience_don | Binaire | Si le candidat a déjà donné du sang (1=Oui, 0=Non) |
| Genre | Catégorielle | Genre du candidat (Homme/Femme) |
| Niveau d'etude | Catégorielle | Niveau d'éducation |
| Situation Matrimoniale (SM) | Catégorielle | État civil |
| Profession | Catégorielle | Métier du candidat |
| Arrondissement de résidence | Catégorielle | Lieu d'habitation |
| Quartier de Résidence | Catégorielle | Quartier spécifique |
| Nationalité | Catégorielle | Nationalité du candidat |
| Religion | Catégorielle | Religion du candidat |
| A-t-il (elle) déjà donné le sang | Binaire | Expérience antérieure de don |
| Taux d'hémoglobine | Numérique | Niveau d'hémoglobine en g/dL |
| ÉLIGIBILITÉ AU DON. | Catégorielle | Variable cible (Eligible, Temporairement Non-eligible, Définitivement non-eligible) |

## Prétraitement des données

### Nettoyage des données

Le prétraitement des données a impliqué les étapes suivantes:

1. **Conversion des formats de date**: Les dates ont été converties en format datetime pour faciliter les calculs.
2. **Gestion des valeurs manquantes**: 
   - Imputation des valeurs numériques manquantes par la médiane
   - Imputation des valeurs catégorielles manquantes par le mode (valeur la plus fréquente)
3. **Filtrage des caractéristiques**: Élimination des colonnes avec plus de 50% de valeurs manquantes.
4. **Standardisation des noms d'arrondissement et de quartier**: Création des variables `arrondissement_clean` et `quartier_clean`.
5. **Création de caractéristiques dérivées**:
   - Calcul de l'âge à partir des dates de naissance
   - Création de groupes d'âge 
   - Conversion de l'éligibilité en variable binaire pour la modélisation

### Encodage des variables

Les variables catégorielles ont été encodées en utilisant:
- **One-Hot Encoding** pour les variables à faible cardinalité (Genre, Niveau d'étude, etc.)
- **Encodage simple** pour les variables binaires

## Exploration et analyse des données

### Distribution de l'éligibilité

La distribution des classes d'éligibilité était la suivante:
- **Eligible**: **[POURCENTAGE]**% (**[NOMBRE]** candidats)
- **Temporairement Non-eligible**: **[POURCENTAGE]**% (**[NOMBRE]** candidats)
- **Définitivement non-eligible**: **[POURCENTAGE]**% (**[NOMBRE]** candidats)

Cette distribution présente un déséquilibre de classes qui a été pris en compte lors de la modélisation.

### Corrélations principales

Les corrélations les plus importantes avec l'éligibilité étaient:
- **Taux d'hémoglobine**: Corrélation positive de **[VALEUR]**
- **Conditions médicales** (VIH, diabète, problèmes cardiaques): Corrélations négatives entre **[VALEUR]** et **[VALEUR]**
- **Âge**: Corrélation de **[VALEUR]**

### Insights clés de l'exploration

1. Les porteurs de VIH ou d'hépatite sont systématiquement non éligibles
2. Un taux d'hémoglobine insuffisant est une cause fréquente d'inéligibilité temporaire
3. Certaines zones géographiques présentent des taux d'éligibilité significativement différents
4. Les donneurs réguliers ont un taux d'éligibilité plus élevé que les nouveaux donneurs

## Développement des modèles

### Approche de modélisation

Pour développer le modèle de prédiction d'éligibilité, nous avons:
1. Convertit le problème en classification binaire (Éligible vs Non éligible)
2. Divisé les données en ensembles d'entraînement (80%) et de test (20%)
3. Créé un pipeline complet incluant:
   - Prétraitement (imputation, standardisation, encodage)
   - Modélisation
   - Évaluation

### Modèles candidats

Trois modèles de classification ont été évalués:

#### 1. Random Forest
- Avantages: Bonne performance générale, interprétabilité, gestion des données non linéaires
- Hyperparamètres optimisés:
  - n_estimators: **[VALEUR]**
  - max_depth: **[VALEUR]**
  - min_samples_split: **[VALEUR]**
  - min_samples_leaf: **[VALEUR]**
  - class_weight: **[VALEUR]**

#### 2. Gradient Boosting
- Avantages: Performance supérieure, bonne gestion du déséquilibre des classes
- Hyperparamètres optimisés:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 5
  - min_samples_split: 2
  - min_samples_leaf: 2
  - subsample: 0.9

#### 3. XGBoost
- Avantages: Performance de pointe, rapidité, robustesse aux valeurs aberrantes
- Hyperparamètres optimisés:
  - n_estimators: **[VALEUR]**
  - learning_rate: **[VALEUR]**
  - max_depth: **[VALEUR]**
  - subsample: **[VALEUR]**
  - colsample_bytree: **[VALEUR]**
  - scale_pos_weight: **[VALEUR]**

### Stratégie d'optimisation

Pour chaque modèle, nous avons utilisé:
- **Validation croisée stratifiée**: 5 plis pour assurer une répartition équilibrée des classes
- **Recherche aléatoire d'hyperparamètres**: Plus efficace qu'une recherche exhaustive
- **Optimisation du F1 score**: Métrique appropriée pour les données déséquilibrées

## Évaluation comparative des modèles

### Métriques d'évaluation

Les modèles ont été évalués sur les métriques suivantes:

| Métrique | Description | Importance |
|----------|-------------|------------|
| Accuracy | Proportion de prédictions correctes | Moins pertinente en cas de déséquilibre |
| Precision | Proportion de vrais positifs parmi les prédictions positives | Critique pour minimiser les faux positifs |
| Recall | Proportion de vrais positifs identifiés | Important pour ne pas manquer d'éligibles |
| F1 Score | Moyenne harmonique de précision et rappel | Métrique équilibrée |
| AUC ROC | Aire sous la courbe ROC | Performance générale |

### Résultats comparatifs

| Modèle | Accuracy | Precision | Recall | F1 Score | AUC ROC | Temps d'entraînement |
|--------|----------|-----------|--------|----------|---------|---------------------|
| Random Forest | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** s |
| Gradient Boosting | 0.8903 | 0.9162 | 0.9563 | 0.9358 | 0.8816 | **[VALEUR]** s |
| XGBoost | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** | **[VALEUR]** s |

### Analyse des erreurs

Analyse des erreurs de prédiction par modèle:

#### Random Forest
- Faux positifs: **[NOMBRE]** cas (**[POURCENTAGE]**%)
- Faux négatifs: **[NOMBRE]** cas (**[POURCENTAGE]**%)
- Principales causes d'erreur: **[DESCRIPTION]**

#### Gradient Boosting
- Faux positifs: **[NOMBRE]** cas (**[POURCENTAGE]**%)
- Faux négatifs: **[NOMBRE]** cas (**[POURCENTAGE]**%)
- Principales causes d'erreur: **[DESCRIPTION]**

#### XGBoost
- Faux positifs: **[NOMBRE]** cas (**[POURCENTAGE]**%)
- Faux négatifs: **[NOMBRE]** cas (**[POURCENTAGE]**%)
- Principales causes d'erreur: **[DESCRIPTION]**

## Modèle final retenu

### Choix du modèle

Après comparaison des performances, nous avons retenu le modèle **Gradient Boosting** pour les raisons suivantes:
1. **Meilleur F1 score** (0.9358) parmi les modèles évalués
2. **Équilibre optimal** entre précision (0.9162) et rappel (0.9563)
3. **Robustesse** face au déséquilibre des classes
4. **Bonne interprétabilité** des caractéristiques importantes

### Caractéristiques importantes

Les caractéristiques les plus importantes pour la prédiction d'éligibilité selon le modèle final sont:

1. **[CARACTÉRISTIQUE 1]**: Importance relative **[VALEUR]**
2. **[CARACTÉRISTIQUE 2]**: Importance relative **[VALEUR]**
3. **[CARACTÉRISTIQUE 3]**: Importance relative **[VALEUR]**
4. **[CARACTÉRISTIQUE 4]**: Importance relative **[VALEUR]**
5. **[CARACTÉRISTIQUE 5]**: Importance relative **[VALEUR]**

### Performance détaillée

Performance détaillée du modèle final:
- **Accuracy**: 0.8903
- **Precision**: 0.9162
- **Recall**: 0.9563
- **F1 Score**: 0.9358
- **AUC ROC**: 0.8816
- **Matrice de confusion**:
  ```
  [MATRICE]
  ```

### Interprétation du modèle

L'analyse des prédictions montre que:
1. **[INSIGHT 1]**
2. **[INSIGHT 2]**
3. **[INSIGHT 3]**

## Règles de sécurité implémentées

Malgré la bonne performance du modèle, nous avons identifié un problème critique: le modèle pouvait parfois prédire comme éligibles des candidats présentant des contre-indications absolues (comme être porteur de VIH ou d'hépatite).

Pour garantir la sécurité transfusionnelle, nous avons implémenté un système de règles strictes qui s'appliquent indépendamment des prédictions du modèle:

### Critères d'exclusion absolus

Les conditions suivantes entraînent une inéligibilité automatique:
1. **Porteur de VIH, hépatite B ou C**: Contre-indication absolue pour protéger les receveurs
2. **Drépanocytaire**: Contre-indication pour la sécurité du donneur
3. **Problèmes cardiaques sévères**: Contre-indication pour la sécurité du donneur

### Vérifications supplémentaires

Des vérifications supplémentaires incluent:
1. **Taux d'hémoglobine minimum**: 12 g/dL pour les femmes, 13 g/dL pour les hommes
2. **Âge**: Entre 18 et 65 ans

### Implémentation technique

Ces règles sont appliquées:
1. **Avant la prédiction** du modèle (vérification préliminaire)
2. **Après la prédiction** comme vérification finale de sécurité
3. **Dans l'interface utilisateur** sous forme d'avertissements visuels

## Intégration au tableau de bord

### Workflow de prédiction

Le modèle a été intégré au tableau de bord avec le workflow suivant:
1. **Saisie des données** du donneur potentiel via l'interface
2. **Normalisation des caractéristiques** pour correspondre au format attendu par le modèle
3. **Application des règles de sécurité** préliminaires
4. **Prédiction** par le modèle d'apprentissage automatique
5. **Vérification finale** des règles de sécurité absolues
6. **Affichage du résultat** avec explication et niveau de confiance

### Fonctionnalités d'interface

L'interface de prédiction inclut:
- Organisation en onglets thématiques pour une saisie claire
- Avertissements visuels pour les critères d'exclusion
- Affichage des facteurs déterminants en cas d'inéligibilité
- Niveau de confiance de la prédiction

## Limites et perspectives d'amélioration

### Limites identifiées

1. **Déséquilibre des données**: Surreprésentation des candidats éligibles
2. **Valeurs manquantes**: Certaines caractéristiques présentent un taux élevé de valeurs manquantes
3. **Couverture géographique limitée**: Principalement des données de la région de Douala
4. **Potentiel surajustement** aux pratiques spécifiques de sélection

### Améliorations futures

1. **Collecte de données supplémentaires** pour améliorer la représentativité
2. **Modèles plus sophistiqués** comme les réseaux de neurones pour capturer des interactions complexes
3. **Analyse temporelle** pour identifier les tendances saisonnières
4. **Modèle multiclasse** pour distinguer l'inéligibilité temporaire de l'inéligibilité définitive
5. **Calibration de probabilité** pour améliorer l'interprétation des scores de confiance

## Conclusion

Le modèle de prédiction d'éligibilité au don de sang basé sur Gradient Boosting offre d'excellentes performances avec un F1 score de 0.9358, une précision de 0.9162 et un rappel de 0.9563. L'intégration de règles de sécurité strictes compense les limites inhérentes aux approches d'apprentissage automatique, garantissant ainsi la sécurité des donneurs et des receveurs.

Cette approche hybride combinant apprentissage automatique et règles métier constitue une solution robuste pour le pré-screening des donneurs potentiels, contribuant à l'optimisation des campagnes de don de sang.