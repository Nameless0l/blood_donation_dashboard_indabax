# Documentation de l'API de prédiction d'éligibilité au don de sang

## Vue d'ensemble

Cette API permet de prédire l'éligibilité d'un donneur potentiel au don de sang en fonction de ses caractéristiques démographiques et médicales. L'API utilise un modèle d'apprentissage automatique entraîné sur des données historiques, complété par des règles strictes de sécurité transfusionnelle.

## Base URL

```
http://localhost:8000
```

Pour un déploiement production, remplacez par l'URL de votre serveur.

## Endpoints

### Prédiction d'éligibilité

```
POST /api/predict_eligibility
```

Prédit si un donneur potentiel est éligible au don de sang.

#### Paramètres de la requête

Le corps de la requête doit être un objet JSON contenant les informations du donneur potentiel :

```json
{
  "donnees_demographiques": {
    "age": 35,
    "genre": "Homme",
    "niveau_etude": "Universitaire",
    "situation_matrimoniale": "Marié(e)",
    "profession": "Ingénieur",
    "nationalite": "Camerounaise",
    "religion": "Chrétien(ne)",
    "experience_don": 1
  },
  "donnees_medicales": {
    "porteur_vih_hbs_hcv": 0,
    "diabetique": 0,
    "hypertendu": 0,
    "asthmatique": 0,
    "drepanocytaire": 0,
    "cardiaque": 0,
    "taux_hemoglobine": 13.5
  },
  "donnees_geographiques": {
    "arrondissement": "Douala 3",
    "quartier": "Logbaba"
  }
}
```

##### Champs obligatoires

- `donnees_demographiques.age` (entier) : Âge du donneur en années (18-70)
- `donnees_demographiques.genre` (chaîne) : "Homme" ou "Femme"
- `donnees_medicales.porteur_vih_hbs_hcv` (entier) : 0 pour Non, 1 pour Oui
- `donnees_medicales.taux_hemoglobine` (décimal) : Taux d'hémoglobine en g/dL

##### Champs optionnels

Tous les autres champs sont optionnels. S'ils ne sont pas fournis, des valeurs par défaut seront utilisées basées sur les statistiques de la population des donneurs.

#### Réponse

```json
{
  "prediction": "Éligible",
  "confidence": 92.5,
  "facteurs_importants": ["Âge", "Taux d'hémoglobine", "Expérience de don"],
  "messages": []
}
```

ou

```json
{
  "prediction": "Non éligible",
  "confidence": 98.7,
  "facteurs_importants": ["Porteur de VIH, hépatite B ou C"],
  "messages": ["Critère d'exclusion absolu détecté"]
}
```

##### Champs de la réponse

- `prediction` (chaîne) : "Éligible" ou "Non éligible"
- `confidence` (décimal) : Pourcentage de confiance dans la prédiction (0-100)
- `facteurs_importants` (tableau de chaînes) : Liste des facteurs qui ont le plus influencé la prédiction
- `messages` (tableau de chaînes) : Messages informatifs ou d'avertissement

#### Codes de statut

- `200 OK` : Requête traitée avec succès
- `400 Bad Request` : Paramètres manquants ou invalides
- `500 Internal Server Error` : Erreur lors du traitement de la requête

### Statut du service

```
GET /api/status
```

Vérifie si l'API est opérationnelle.

#### Réponse

```json
{
  "status": "ok",
  "model_version": "gradient_boosting_20250323_104955",
  "uptime": "3h 24m 15s"
}
```

## Exemples d'utilisation

### Exemple 1 : Donneur éligible

**Requête :**

```bash
curl -X POST "http://localhost:8000/api/predict_eligibility" \
     -H "Content-Type: application/json" \
     -d '{
           "donnees_demographiques": {
             "age": 35,
             "genre": "Homme",
             "experience_don": 1
           },
           "donnees_medicales": {
             "porteur_vih_hbs_hcv": 0,
             "taux_hemoglobine": 14.2
           },
           "donnees_geographiques": {
             "arrondissement": "Douala 3"
           }
         }'
```

**Réponse :**

```json
{
  "prediction": "Éligible",
  "confidence": 92.5,
  "facteurs_importants": ["Âge", "Taux d'hémoglobine", "Expérience de don"],
  "messages": []
}
```

### Exemple 2 : Donneur non éligible (porteur de VIH)

**Requête :**

```bash
curl -X POST "http://localhost:8000/api/predict_eligibility" \
     -H "Content-Type: application/json" \
     -d '{
           "donnees_demographiques": {
             "age": 27,
             "genre": "Femme"
           },
           "donnees_medicales": {
             "porteur_vih_hbs_hcv": 1,
             "taux_hemoglobine": 13.0
           }
         }'
```

**Réponse :**

```json
{
  "prediction": "Non éligible",
  "confidence": 100.0,
  "facteurs_importants": ["Porteur de VIH, hépatite B ou C"],
  "messages": ["Critère d'exclusion absolu détecté"]
}
```

### Exemple 3 : Donneur non éligible (taux d'hémoglobine bas)

**Requête :**

```bash
curl -X POST "http://localhost:8000/api/predict_eligibility" \
     -H "Content-Type: application/json" \
     -d '{
           "donnees_demographiques": {
             "age": 42,
             "genre": "Femme"
           },
           "donnees_medicales": {
             "porteur_vih_hbs_hcv": 0,
             "taux_hemoglobine": 10.5
           }
         }'
```

**Réponse :**

```json
{
  "prediction": "Non éligible",
  "confidence": 87.3,
  "facteurs_importants": ["Taux d'hémoglobine bas"],
  "messages": ["Taux d'hémoglobine insuffisant pour une femme (minimum 12.0 g/dL)"]
}
```

## Règles strictes de sécurité

L'API applique des règles strictes de sécurité transfusionnelle, indépendamment des prédictions du modèle d'apprentissage automatique. Les critères d'exclusion absolus suivants rendent automatiquement un donneur non éligible :

1. Porteur de VIH, hépatite B ou C
2. Drépanocytaire
3. Problèmes cardiaques graves
4. Taux d'hémoglobine inférieur à 13 g/dL pour les hommes ou 12 g/dL pour les femmes

## Authentification

L'API nécessite une authentification par clé API dans les en-têtes HTTP :

```
Authorization: Bearer YOUR_API_KEY
```

Contactez l'administrateur du système pour obtenir une clé API valide.

## Limitations

- Maximum de 100 requêtes par minute par clé API
- Taille maximale du corps de la requête : 10 KB

## Intégration

### Intégration avec Python

```python
import requests
import json

url = "http://localhost:8000/api/predict_eligibility"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}

data = {
    "donnees_demographiques": {
        "age": 35,
        "genre": "Homme",
        "experience_don": 1
    },
    "donnees_medicales": {
        "porteur_vih_hbs_hcv": 0,
        "taux_hemoglobine": 14.2
    },
    "donnees_geographiques": {
        "arrondissement": "Douala 3"
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Prédiction : {result['prediction']}")
print(f"Confiance : {result['confidence']}%")
```

### Intégration avec JavaScript

```javascript
async function checkEligibility() {
    const url = "http://localhost:8000/api/predict_eligibility";
    
    const data = {
        donnees_demographiques: {
            age: 35,
            genre: "Homme",
            experience_don: 1
        },
        donnees_medicales: {
            porteur_vih_hbs_hcv: 0,
            taux_hemoglobine: 14.2
        },
        donnees_geographiques: {
            arrondissement: "Douala 3"
        }
    };
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_KEY'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        console.log(`Prédiction : ${result.prediction}`);
        console.log(`Confiance : ${result.confidence}%`);
        
        return result;
    } catch (error) {
        console.error("Erreur lors de la vérification d'éligibilité:", error);
        return null;
    }
}
```

## Support et contact

Pour toute question ou problème concernant l'API, veuillez contacter :

- Email : aron.mbassi@enspy-uy1.cm
- Téléphone : +237 656 820 591

## Changements et mises à jour

### Version 1.0.0 (23 mars 2025)
- Version initiale de l'API