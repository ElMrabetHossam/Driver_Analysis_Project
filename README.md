# Driver_Analysis_Project

# 🚗 Driver Behavior Analysis System

## 📋 Présentation du Projet
Ce projet vise à développer un système complet d'analyse du comportement du conducteur en utilisant une architecture en cascade :
1. **Traitement d'image (Computer Vision)** : Extraction de données depuis la vidéo (détection véhicules, lignes).
2. **Machine Learning** : Analyse des données pour classifier la conduite (Sûre vs Dangereuse).

Le but final est de générer un "Score de Sécurité" et un rapport automatisé.

## 📂 Structure du Projet
Nous devons **strictement** respecter cette architecture pour faciliter la fusion de nos travaux :

```text
Driver_Analysis_Project/
├── data/
│   ├── raw/          # Vidéos MP4 et logs capteurs bruts (Comma2k19)
│   └── processed/    # Fichiers CSV générés après extraction des features
├── src/
├── ├──main.py                  # Pipeline principal
│   ├── image_processing/
│   │   ├── vehicle_tracker.py  # YOLO logic
│   │   └── lane_detector.py    # OpenCV logic
│   ├── features/               # Scripts de fusion (Video + Capteurs)
│   └── models/
│       ├── traditional_ml.py   # SVM, RF, KMeans
│       └── deep_learning.py    # LSTM, Transformer
├── notebooks/        # Pour vos tests et exploration (EDA)
└── requirements.txt  # Liste des dépendances