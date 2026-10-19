# 🧠 BrainScan AI - Détection et Classification des Tumeurs Cérébrales par IRM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **BrainScan AI** est une initiative de startup marocaine innovante spécialisée dans les technologies d'imagerie médicale assistée par l'intelligence artificielle. Notre mission est d'optimiser le diagnostic précoce des tumeurs cérébrales grâce à des réseaux convolutifs (CNN) profonds, en réduisant les délais d'attente et en assistant les praticiens dans leurs prises de décision clinique.

---

## 📌 Fonctionnalités Clés

- **Prétraitement & Nettoyage Robuste** :
  - Validation automatique des formats (`.jpeg`, `.jpg`, `.bmp`, `.png`) et suppression des fichiers invalides.
  - Gestion sécurisée des erreurs de lecture via blocs `try-except`.
  - Redimensionnement standardisé à **224×224 px** avec OpenCV.
  - Mélange synchronisé des listes d'images et d'étiquettes.
  - Normalisation des intensités de pixels dans l'intervalle `[0, 1]`.
- **Analyse Exploratoire des Données (EDA)** :
  - Visualisation de la distribution des classes et contrôle du rééquilibrage.
  - Grille d'échantillons représentatifs par pathologie.
- **Modélisation CNN Optimisée** :
  - Empilement de blocs convolutifs `Conv2D`, `MaxPooling2D`, `BatchNormalization`, et régularisation `Dropout`.
  - Optimiseur Adam et fonction de perte `categorical_crossentropy`.
  - Callbacks intelligents : `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
  - Suivi précis du temps d'entraînement (`time`).
- **Évaluation Médicale Complète** :
  - Courbes d'apprentissage (`accuracy` et `loss`).
  - Matrice de confusion détaillée (`Seaborn Heatmap`).
  - Rapport de classification clinique (Précision, Rappel, F1-Score).
  - Échantillonnage comparatif des prédictions correctes vs anomalies.
- **Inférence Rapide & Application Web Streamlit** :
  - Fonction unitaire `predict_image(path)`.
  - Interface web moderne avec drag & drop d'IRM, indicateur de confiance, et graphique de probabilités.

---

## 📁 Arborescence du Projet

```text
brain_scan/
├── data/
│   ├── raw/                   # Déposer ici les images IRM par classe
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   └── processed/             # Données prétraitées
├── models/                    # Modèles entraînés (.h5) et labels JSON
├── notebooks/                 # Notebook d'expérimentation pas-à-pas
│   └── 01_brain_tumor_mri_pipeline.ipynb
├── reports/
│   └── figures/               # Graphiques générés (courbes, matrices)
├── src/                       # Code source modulaire
│   ├── __init__.py
│   ├── config.py              # Configuration et constantes
│   ├── data_loader.py         # Chargement, redimensionnement & encodage
│   ├── visualization.py       # Fonctions graphiques
│   ├── model.py               # Architecture CNN
│   ├── train.py               # Pipeline d'entraînement
│   ├── evaluate.py            # Évaluation et métriques
│   └── predict.py             # Fonction predict_image
├── app/
│   └── app.py                 # Interface web Streamlit
├── requirements.txt           # Dépendances Python
├── .gitignore
├── TODO.md                    # Feuille de route détaillée
└── README.md
```

---

## 🚀 Installation & Démarrage

### 1. Cloner le projet et créer un environnement virtuel

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activation (Linux / macOS)
source venv/bin/activate
```

### 2. Installer les dépendances requises

```bash
pip install -r requirements.txt
```

### 3. Préparer les données

Déposez vos images IRM dans le dossier `data/raw/` en créant un sous-dossier par classe :
- `data/raw/glioma/`
- `data/raw/meningioma/`
- `data/raw/no_tumor/`
- `data/raw/pituitary/`

---

## 💻 Utilisation des Modules

### 🏋️ Entraîner le Modèle CNN
```bash
python -m src.train
```
*Le meilleur modèle sera automatiquement enregistré sous `models/best_brain_tumor_model.h5` et les métriques sous `reports/figures/`.*

### 📊 Évaluer le Modèle
```bash
python -m src.evaluate
```

### 🔍 Prédire sur un cliché unitaire
```bash
python -m src.predict "chemin/vers/votre/irm.jpg"
```

### 🌐 Lancer l'Application Web Médicale (Streamlit)
```bash
streamlit run app/app.py
```

---

