# Analyse du Comportement Conducteur - Résumé du Projet

> Système complet de classification du comportement conducteur utilisant le dataset Comma2k19 avec vision par ordinateur, ML traditionnel et modèles deep learning.

---

## 🚀 Démarrage Rapide (Test Professeur)

> **⚠️ Important** : Veuillez lire la section [Architecture Pipeline en Deux Étapes](#architecture-pipeline-en-deux-étapes) pour comprendre pourquoi l'extraction de features prend ~40 minutes tandis que l'entraînement des modèles prend <5 minutes.

### Données Échantillon Incluses

| Élément | Taille | Chemin |
|---------|--------|--------|
| **Segment Vidéo** | 50 Mo | `data/raw/comma2k19/Chunk_1/.../10/` |
| **Features Pré-extraites** | 14.5 Mo | `data/processed/training_data.csv` |

### Test Vidéo Démo
```bash
python3 src/demo_generator.py \
    --input "data/raw/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/10" \
    --output demo.mp4 \
    --preview
```

### Dashboard Interactif
```bash
streamlit run src/dashboard.py
```

---

## Table des Matières
1. [Vue d'Ensemble](#vue-densemble)
2. [Dataset & Traitement](#dataset--traitement)
3. [Ingénierie des Features](#ingénierie-des-features)
4. [Performance des Modèles](#performance-des-modèles)
5. [Structure du Projet](#structure-du-projet)
6. [Guide d'Utilisation](#guide-dutilisation)
7. [Résultats & Insights](#résultats--insights)

---

## Vue d'Ensemble

Ce projet implémente un pipeline complet d'analyse du comportement conducteur qui :
- Traite les vidéos de caméra embarquée et les données télémétrie
- Extrait des features visuelles via la détection d'objets YOLO
- Classifie le comportement en **sûr**, **agressif** ou **somnolent**
- Calcule un score de sécurité conducteur (0-100)

### Stack Technologique
- **Vision par Ordinateur** : OpenCV, YOLOv8
- **ML Traditionnel** : scikit-learn (SVM, Random Forest, K-Means, Isolation Forest)
- **Deep Learning** : PyTorch (LSTM, Transformer)
- **Visualisation** : Streamlit, Plotly

---

## Dataset & Traitement

### À Propos de Comma2k19

Le dataset **Comma2k19** est un dataset de conduite à grande échelle publié par [Comma.ai](https://comma.ai). Il contient **33 heures de conduite sur autoroute** enregistrées en Californie, USA.

| Attribut | Détails |
|----------|---------|
| **Taille Totale** | ~100 Go |
| **Durée** | 33 heures de conduite |
| **Segments** | 2,019 segments × ~1 minute chacun |
| **Chunks** | 10 chunks (~10 Go chacun) |
| **Véhicules** | Toyota RAV4 (Chunks 1-2), Honda Civic (Chunks 3-10) |
| **Fréquence** | Vidéo : 20 FPS, Capteurs : 100 Hz |
| **Résolution** | 1164 × 874 pixels (caméra grand-angle) |

### Suite de Capteurs Comma2k19

Chaque segment contient des données capteurs multi-modales synchronisées :

| Capteur | Type | Fréquence | Format | Description |
|---------|------|-----------|--------|-------------|
| **Vidéo** | Caméra | 20 FPS | `.hevc` | Caméra embarquée grand-angle |
| **Vitesse** | CAN Bus | 100 Hz | `.npy` | Vitesse véhicule OBD-II (m/s) |
| **Angle Volant** | CAN Bus | 100 Hz | `.npy` | Angle du volant (degrés) |
| **Accéléromètre** | IMU | 100 Hz | `.npy` | Accélération 3 axes [x,y,z] (m/s²) |
| **Gyroscope** | IMU | 100 Hz | `.npy` | Vitesse angulaire [roll,pitch,yaw] (rad/s) |
| **Radar** | Radar | 20 Hz | `.npy` | Distance véhicule avant & vitesse relative |
| **GPS** | GPS | - | `.npy` | Coordonnées ECEF [x,y,z] |

### Notre Périmètre de Traitement

| Métrique | Valeur |
|----------|--------|
| **Chunk Utilisé** | Chunk 1 (Toyota RAV4) |
| **Segments Traités** | 188 segments |
| **Total Échantillons** | 44,985 points de données labélisés |
| **Fichier de Sortie** | `data/processed/training_data.csv` (14.5 Mo) |

---

## Architecture Pipeline en Deux Étapes

Notre projet utilise une **architecture en deux étapes** qui sépare l'extraction de features de l'entraînement des modèles :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ÉTAPE 1 : EXTRACTION DE FEATURES                      │
│                    (Coûteuse en Calcul - GPU/CPU Intensif)               │
│                           ~37 MINUTES (1 Chunk)                          │
│                        ~7 HEURES (Dataset Complet)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Données Brutes Comma2k19 (Vidéo + Capteurs)                           │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────────┐    ┌──────────────────┐                          │
│   │  Pipeline OpenCV │    │  YOLOv8 (Pré-    │                          │
│   │  Perspective +   │    │  entraîné)       │                          │
│   │  Sliding Window  │    │  → nb_véhicules  │                          │
│   └────────┬─────────┘    │  → distance_lead │                          │
│            │              └────────┬─────────┘                          │
│            │                       │                                     │
│            └───────────┬───────────┘                                     │
│                        ▼                                                 │
│   ┌──────────────────────────────────────────┐                          │
│   │  + Données Capteurs (vitesse, volant,    │                          │
│   │    accel, gyro, radar) + Features        │                          │
│   │    Dérivées (speed_change, jerk)         │                          │
│   └──────────────────────────────────────────┘                          │
│                        │                                                 │
│                        ▼                                                 │
│              training_data.csv (44,985 × 18 features)                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ÉTAPE 2 : ENTRAÎNEMENT DES MODÈLES                    │
│                    (Rapide - ML Tabulaire sur CSV)                       │
│                           ~3 MINUTES                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   training_data.csv                                                      │
│          │                                                               │
│          ├──────────────────┬─────────────────┬───────────────┐         │
│          ▼                  ▼                 ▼               ▼         │
│   ┌────────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐   │
│   │    SVM     │    │   Random    │    │   LSTM   │    │Transformer│   │
│   │  21.9 sec  │    │   Forest    │    │  82 sec  │    │  78 sec   │   │
│   │  77.4%     │    │   0.9 sec   │    │  94.1%   │    │  96.9%    │   │
│   └────────────┘    │   96.3%     │    └──────────┘    └───────────┘   │
│                     └─────────────┘                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Répartition Détaillée des Temps

#### Étape 1 : Extraction de Features

| Tâche | Temps | Notes |
|-------|-------|-------|
| Décodage vidéo (188 segments) | ~5 min | Décodage HEVC avec OpenCV |
| Inférence YOLO (par frame) | ~22 min | YOLOv8n sur chaque frame |
| Détection de voies OpenCV | ~5.5 min | Perspective + Sliding Window |
| Synchronisation capteurs | ~3.7 min | Interpolation aux temps frames |
| Labeling + export CSV | ~1 min | Règles heuristiques + I/O |
| **Total Étape 1** | **~37 min** | **Pour 1 Chunk (188 segments)** |

> **⚠️ Contexte Scalabilité** : 37 minutes représente le traitement d'**un seul chunk** (Chunk 1 : 188 segments).
> Le dataset complet Comma2k19 contient **10 chunks** (~2,019 segments).
> **Temps estimé pour le dataset complet : ~7 HEURES**

#### Étape 2 : Entraînement des Modèles

| Modèle | Temps | Notes |
|--------|-------|-------|
| SVM | 21.95 sec | Noyau RBF, 36K échantillons |
| Random Forest | 0.90 sec | 100 arbres, parallélisé |
| K-Means | 7.66 sec | 3 clusters |
| Isolation Forest | 0.22 sec | Détection d'anomalies |
| LSTM | 82.12 sec | 15 époques, GPU MPS |
| Transformer | 77.55 sec | 15 époques, GPU MPS |
| **Total Étape 2** | **~3.2 min** | **Entraînement réel mesuré** |

> **Pourquoi l'Étape 2 est-elle si rapide ?**
> - Nous entraînons sur des **données tabulaires** (18 features numériques), pas sur des images brutes
> - YOLO est **pré-entraîné** — nous l'utilisons uniquement pour l'inférence
> - **Apple Silicon (M2)** : L'entraînement deep learning utilise le GPU via Metal Performance Shaders (MPS), ce qui accélère significativement les calculs PyTorch
> - Cette conception en deux étapes est efficace et permet une itération rapide des modèles

### Distribution des Labels

| Label | Nombre | Pourcentage | Description |
|-------|--------|-------------|-------------|
| **Agressif** | 27,582 | 61.3% | Freinage brusque, talonnage, volant soudain |
| **Sûr** | 16,017 | 35.6% | Conduite normale dans les seuils |
| **Somnolent** | 1,386 | 3.1% | Déviation de voie, volant incohérent |

---

## Ingénierie des Features

### Features Extraites (18 au total)

| Feature | Source | Description |
|---------|--------|-------------|
| `speed` | CAN bus | Vitesse véhicule (m/s) |
| `steering` | CAN bus | Angle du volant (degrés) |
| `accel_forward` | IMU | Accélération avant (m/s²) |
| `accel_lateral` | IMU | Accélération latérale (m/s²) |
| `accel_vertical` | IMU | Accélération verticale (m/s²) |
| `gyro_yaw` | IMU | Taux de lacet (rad/s) |
| `radar_distance` | Radar | Distance au véhicule avant (m) |
| `radar_rel_speed` | Radar | Vitesse relative véhicule avant (m/s) |
| `vehicle_count` | YOLO | Nombre de véhicules détectés |
| `lead_distance_visual` | YOLO | Estimation visuelle distance avant |
| `speed_change` | Dérivé | Accélération depuis vitesse (m/s²) |
| `steering_rate` | Dérivé | Taux de changement volant (deg/s) |
| `steering_jerk` | Dérivé | Jerk du volant (deg/s²) |

### Seuils de Labeling

Le labeler heuristique classifie le comportement selon :

```python
AGRESSIF si :
    - steering_jerk > 5.0 deg/s²
    - speed_change < -3.0 m/s² (freinage brusque)
    - radar_distance < 15m (talonnage)

SOMNOLENT si :
    - lane_deviation > 0.5m (zigzag)
    - steering_rate constamment faible
    - variation de vitesse minimale

SÛR : sinon
```

---

## Performance des Modèles

### Résultats Finaux (Entraînés sur 44,985 échantillons)

| Modèle | Accuracy | F1 Score | Type |
|--------|----------|----------|------|
| SVM | 77.4% | 80.7% | ML Traditionnel |
| **Random Forest** | **96.3%** | **96.3%** | ML Traditionnel |
| LSTM | 94.1% | 93.8% | Deep Learning |
| **Transformer** | **96.9%** | **96.6%** | Deep Learning |

> **Meilleur Modèle** : Transformer avec 96.9% accuracy

### Importance des Features (Random Forest)

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | `steering_jerk` | 35.4% |
| 2 | `speed` | 29.2% |
| 3 | `steering` | 8.8% |
| 4 | `radar_distance` | 6.3% |
| 5 | `steering_rate` | 4.5% |

### Résultats Apprentissage Non-Supervisé

**K-Means Clustering** (3 clusters) :
- Score Silhouette : 0.178
- Découverte de 3 styles de conduite distincts

**Isolation Forest** (Détection d'Anomalies) :
- Contamination : 10%
- Précision sur conduite agressive : 80%

---

## Structure du Projet

```
Driver_Analysis_Project/
├── data/
│   ├── raw/
│   │   └── comma2k19/
│   │       └── Chunk_1/           # 188 segments de conduite
│   └── processed/
│       └── training_data.csv      # Données d'entraînement principales
│
├── models/
│   ├── random_forest_model.pkl    # Meilleur ML traditionnel (8.8 Mo)
│   ├── svm_model.pkl              # Classifieur SVM (1.6 Mo)
│   ├── lstm_model.pt              # Modèle LSTM (549 Ko)
│   └── transformer_model.pt       # Meilleur global (303 Ko)
│
├── src/
│   ├── features/
│   │   ├── data_loader.py         # Chargement données Comma2k19
│   │   ├── feature_extractor.py   # Extraction YOLO + télémétrie
│   │   ├── synchronizer.py        # Synchronisation temporelle
│   │   ├── labeler.py             # Labeling heuristique
│   │   ├── batch_process.py       # Traitement multi-segments
│   │   └── download_data.py       # Helper téléchargement dataset
│   │
│   ├── models/
│   │   ├── traditional_ml.py      # SVM, RF, K-Means, IsoForest
│   │   ├── deep_learning.py       # LSTM, Transformer
│   │   ├── scorer.py              # Calcul score de sécurité
│   │   └── report_generator.py    # Génération rapports PDF
│   │
│   ├── image_processing/
│   │   ├── lane_detector.py       # Détection voies (Perspective + Sliding Window)
│   │   ├── vehicle_tracker.py     # Suivi véhicules YOLO + boîtes 3D
│   │   ├── driver_monitor.py      # Moniteur conduite avec HUD
│   │   ├── environment_scanner.py # Détection environnement
│   │   └── report_generator.py    # Génération rapport PNG
│   │
│   ├── demo_generator.py          # Générateur vidéo démo ADAS
│   └── dashboard.py               # Dashboard Streamlit
│
└── requirements.txt               # Dépendances Python
```

---

## Guide d'Utilisation

### Prérequis

```bash
# Installer les dépendances
pip install -r requirements.txt
```

### Commandes Disponibles

> **⚠️ Légende** : ✅ = Fonctionne avec l'échantillon inclus | ⚠️ = Nécessite le dataset complet (~10 Go)

| Commande | Description | Données Requises |
|----------|-------------|------------------|
| **Vidéo Démo** | `python3 src/demo_generator.py --input "..." --output demo.mp4` | ✅ Échantillon |
| **Dashboard** | `streamlit run src/dashboard.py` | ✅ Échantillon |
| **Entraîner ML** | `python3 -m src.models.traditional_ml --train ...` | ✅ Échantillon |
| **Entraîner DL** | `python3 -m src.models.deep_learning --train ...` | ✅ Échantillon |
| **Score Sécurité** | `python3 -m src.models.scorer --input ...` | ✅ Échantillon |
| **Rapport PDF** | `python3 -m src.models.report_generator --input ...` | ✅ Échantillon |
| **Extraction Features** | `python3 -m src.features.batch_process --num-segments 188` | ⚠️ Dataset complet |

### 1. Entraîner les Modèles ML Traditionnels

```bash
python3 -m src.models.traditional_ml \
    --train data/processed/training_data.csv \
    --output models/

# Sortie :
# - models/svm_model.pkl
# - models/random_forest_model.pkl
```

### 2. Entraîner les Modèles Deep Learning

```bash
python3 -m src.models.deep_learning \
    --train data/processed/training_data.csv \
    --output models/

# Sortie :
# - models/lstm_model.pt
# - models/transformer_model.pt
```

### 3. Calculer le Score de Sécurité

```bash
python3 -m src.models.scorer \
    --input data/processed/training_data.csv

# Sortie :
# SCORE GLOBAL : 67.1/100
# NOTE : D
# RISQUE : RISQUE MODÉRÉ
```

### 4. Lancer le Dashboard

```bash
streamlit run src/dashboard.py
# Ouvre http://localhost:8501
```

### 5. Générer une Vidéo Démo

```bash
python3 src/demo_generator.py \
    --input "data/raw/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/10" \
    --output demo.mp4 \
    --preview
# Crée : demo.mp4 + data/processed/demo_report.png
```

---

## Résultats & Insights

### Découvertes Clés

1. **Le jerk du volant est la feature la plus prédictive** (34% importance)
   - Les changements brusques de volant indiquent fortement une conduite agressive

2. **Le deep learning surpasse légèrement le ML traditionnel**
   - Transformer : 96.9% vs Random Forest : 96.6%
   - Mais Random Forest est plus rapide à entraîner et interpréter

3. **Déséquilibre des labels**
   - 62% agressif, 35% sûr, 3% somnolent
   - Les seuils heuristiques peuvent nécessiter un ajustement

4. **Données autoroutières avec détection de voies limitée**
   - Les marquages de voie souvent effacés/peu clairs
   - La distance radar est plus fiable pour le comportement de suivi

### Composants du Score de Sécurité

| Composant | Poids | Description |
|-----------|-------|-------------|
| Comportement | 40% | Classification ML |
| Fluidité | 25% | Jerk volant/accélération |
| Conscience | 20% | Distance de suivi |
| Vitesse | 15% | Conformité vitesse |

### Exemple de Sortie Score

```
========================================
  SCORE GLOBAL : 67.1/100
  NOTE : D
  RISQUE : RISQUE MODÉRÉ
========================================

--- Détail du Score ---
  Comportement :  49.0/100
  Fluidité :      58.8/100
  Conscience :    89.2/100
  Vitesse :       100.0/100

--- Facteurs de Risque ---
  ⚠️  Volant brusque fréquent (60% du temps)

--- Recommandations ---
  → Faire une pause - signes de conduite agressive détectés
  → Pratiquer un volant et une accélération plus fluides
```

---

## Cadre de Décision Contractuelle

Basé sur le score de sécurité, les conducteurs sont automatiquement classifiés :

| Plage Score | Décision | Action |
|-------------|----------|--------|
| **> 85** | 🟢 **ÉLIGIBLE BONUS** | Prime de performance, reconnaissance |
| **70-85** | 🔵 **MAINTIEN CONTRAT** | Renouvellement standard |
| **50-70** | 🟠 **FORMATION OBLIGATOIRE** | Formation sécurité, révision 30 jours |
| **< 50** | 🔴 **RÉSILIATION CONTRAT** | Révision RH, résiliation contrat |

### Générer un Rapport PDF

```bash
python3 -m src.models.report_generator \
    --input data/processed/training_data.csv \
    --driver-id "DRV_2024_001" \
    --driver-name "Jean Dupont" \
    --output reports/
```

Le rapport PDF inclut :
- Résumé exécutif avec score et note
- Recommandation contractuelle avec actions requises
- Détail du score avec visualisations
- Facteurs de risque et recommandations sécurité
- Lignes de signature pour approbation management

---

## Améliorations Futures

1. **Télécharger les chunks supplémentaires** (Chunks 2-10 disponibles)
2. **Affiner les seuils de labeling** avec expertise métier
3. **Ajouter l'inférence temps réel** depuis flux caméra
4. **Déployer le dashboard** sur le cloud (Streamlit Cloud, Heroku)
