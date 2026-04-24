# 🛒 Walmart Weekly Sales Forecast

> Système de prévision des ventes hebdomadaires par département et par boutique,
> basé sur un modèle **XGBoost** optimisé et une application interactive **Streamlit**.

---

## 📌 Contexte

Ce projet s'inscrit dans le cadre de notre projet de fin d'étude en sciences des données au sein de la Cité collégiale
Le jeu de données provient de Kaggle via le lien suivant
https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast.
L'objectif est de concevoir un système intelligent capable de **prévoir les ventes
hebdomadaires** de chaque département au sein des magasins Walmart, en exploitant
les données historiques, les patterns saisonniers, les promotions et les indicateurs
macroéconomiques.

Les données couvrent **45 boutiques** et **jusqu'à 99 départements** par boutique,
sur la période de **février 2010 à octobre 2012**.

---

## 🗂️ Structure du Projet

```
Walmart-Project/
│
├── data/                         # Données brutes (non versionnées)
│   ├── train.csv                 # Historique des ventes
│   ├── test.csv                  # Semaines à prédire
│   ├── stores.csv                # Type et taille des boutiques
│   └── features.csv              # Variables macroéconomiques et promotions
│
├── model_artifacts/              # Artefacts du modèle entraîné
│   ├── xgb_model.pkl             # Modèle XGBoost optimisé
│   ├── features.pkl              # Liste des features
│   ├── label_encoder_type.pkl    # Encodeur du type de boutique
│   ├── target_encoding.pkl       # Target encoding Store × Dept
│   ├── holidays.json             # Dates des fêtes américaines
│   ├── historique_ventes.json    # Historique pour le calcul des lags
│   └── holdout_semaine.json      # Semaine réelle 2012-10-26 (évaluation)
│
├── projet.ipynb  # Notebook complet (EDA → Modélisation → Déploiement)
├── app.py                        # Application Streamlit
├── requirements.txt              # Dépendances Python
└── README.md
```

---

## 🔬 Démarche & Méthodologie

### 1. Exploration des Données (EDA)
- Analyse des valeurs manquantes dans les colonnes de promotions (MarkDown) —
  interprétées comme une **absence de promotion** et remplacées par 0
- Détection et flagging des **ventes négatives ou nulles** (`Is_Vente`)
- Identification des **4 fêtes américaines** ayant un impact sur les ventes :
  Super Bowl, Labor Day, Thanksgiving et Noël
- Analyse de l'impact des fêtes par type de boutique (A, B, C) —
  **Noël** est la semaine la plus profitable pour type de boutique 
- Test statistique (Z-test + OLS) confirmant l'**impact significatif des promotions**

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `Lag_1`, `Lag_2`, `Lag_4` | Ventes des semaines précédentes |
| `Rolling_mean_4/8` | Moyenne glissante sur 4 et 8 semaines |
| `Rolling_std_4/8` | Écart-type glissant (volatilité) |
| `Week_sin/cos`, `Month_sin/cos` | Encodage cyclique du temps |
| `Is_SuperBowl`, `Is_LaborDay`, `Is_Chrismas`, `Is_Thankgiving` | Indicateurs de fêtes |
| `Promotion_Active`, `Nombre_Promotion` | Activité promotionnelle |
| `Store_Dept_Encoded` | Target encoding Store × Département |

### 3. Modélisation
Quatre modèles ont été entraînés et comparés sur un **split temporel (80/20)** :

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | - | - | - |
| **XGBoost** ✅ | **Meilleur** | **Meilleur** | **Meilleur** |
| CNN 1D | - | - | - |
| LSTM | - | - | - |

> Le modèle **XGBoost** a été retenu comme meilleur modèle et optimisé via
> `RandomizedSearchCV` avec validation temporelle (`TimeSeriesSplit`).

### 4. Interprétabilité — XAI avec SHAP
- **Bar Plot** : importance globale des features
- **Beeswarm Plot** : distribution des impacts SHAP
- **Waterfall Plot** : explication locale d'une prédiction individuelle
- **Dependence Plot** : relation entre la feature dominante et son impact SHAP

Les variables de **lag** (`Lag_1`, `Lag_2`, `Lag_4`) et le `Store_Dept_Encoded`
dominent l'importance globale, confirmant que **l'historique récent est le
meilleur prédicteur des ventes futures**.

---

##  Application Streamlit

L'application propose deux modes d'utilisation :

###  Mode Prédiction Libre
- Sélection d'une boutique (aléatoire ou manuelle)
- Tirage de 6 départements (aléatoire ou manuel)
- Choix libre de la semaine à prédire
- Affichage des KPIs, tableau de résultats et graphiques

###  Mode Évaluation Hold-out
- La semaine du **2012-10-26** a été isolée **avant l'entraînement** du modèle
  (jamais vue lors de la phase d'apprentissage)
- Comparaison **Prédit vs Réel** département par département
- Métriques de performance : MAE, RMSE, MAPE, R²
- Jauge de qualité automatique (✅ Excellent / ⚠️ Acceptable / ❌ Insuffisant)

>  L'historique des **2 derniers mois** du fichier d'entraînement est conservé
> en mémoire pour calculer les variables de lag — uniquement pour la boutique
> sélectionnée, afin d'éviter toute fuite de données inter-boutiques.

---

##  Installation & Lancement

### Prérequis
- Python 3.9+
- Les fichiers CSV dans `data/` et les artefacts dans `model_artifacts/`

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/walmart-sales-forecast.git
cd walmart-sales-forecast

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

### requirements.txt
```
streamlit
pandas
numpy
xgboost
scikit-learn
```

---

## 📁 Données

Les données sont disponibles sur Kaggle :
👉 [Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)

Téléchargez les fichiers `train.csv`, `test.csv`, `stores.csv` et `features.csv`
et placez-les dans le dossier `data/`.

>  Les données ne sont pas versionnées dans ce dépôt conformément aux
> conditions d'utilisation de Kaggle.

---

## 🔑 Points Clés Techniques

**Hold-out temporel strict** — La dernière semaine (`2012-10-26`) est isolée
avant tout feature engineering afin de garantir une évaluation réaliste du modèle
sur des données qu'il n'a jamais vues.

**Pas de data leakage** — Les variables de lag et les rolling means sont calculées
avec un décalage (`shift(1)`) et le target encoding est estimé exclusivement sur
les données d'entraînement avant d'être appliqué au jeu de validation.

**Mémoire légère** — L'application ne conserve que 2 mois d'historique pour
calculer les lags, ce qui limite l'empreinte mémoire tout en couvrant la fenêtre
nécessaire (lag max = 4 semaines).

---

## 👤 Auteur

Projet réalisé par:
1. Brice Yakam Yakam 
2. Magloire Dakeyo 
3. Viny Steve Hapi Sandjong
4. Fadimatou Traoré 
5. Toussaint Arakaza
6. Robleh Abdillahi Omar

.
