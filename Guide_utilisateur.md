# 📖 Guide Utilisateur — Walmart Sales Forecast

> Application de prévision des ventes hebdomadaires par département et par boutique,
> alimentée par un modèle **XGBoost** et une interface interactive **Streamlit**.

---

## Table des matières

1. [Présentation de l'application](#1-présentation-de-lapplication)
2. [Prérequis et installation](#2-prérequis-et-installation)
3. [Lancement de l'application](#3-lancement-de-lapplication)
4. [Interface — Vue d'ensemble](#4-interface--vue-densemble)
5. [Mode Prédiction Libre](#5-mode-prédiction-libre--guide-pas-à-pas)
6. [Mode Évaluation Hold-out](#6-mode-évaluation-hold-out--guide-pas-à-pas)
7. [Comprendre les résultats](#7-comprendre-les-résultats)
8. [Questions fréquentes (FAQ)](#8-questions-fréquentes-faq)
9. [Problèmes connus et solutions](#9-problèmes-connus-et-solutions)

---

## 1. Présentation de l'application

L'application **Walmart Sales Forecast** permet de prédire les ventes hebdomadaires
de n'importe quel département au sein des magasins Walmart.

Elle s'adresse à toute personne souhaitant :

- **Explorer** les prévisions de ventes sans écrire une seule ligne de code
- **Évaluer** la performance du modèle sur une semaine réelle jamais utilisée
  lors de l'entraînement
- **Comprendre** l'impact des variables historiques (lags, tendances) sur les
  prédictions

L'application propose **deux modes d'utilisation** :

| Mode | Description | Usage recommandé |
|---|---|---|
| **Prédiction libre** | Choisissez librement boutique, départements et semaine | Exploration, démonstration |
| **Évaluation Hold-out** | Comparez prédit vs réel sur la semaine du 2012-10-26 | Validation du modèle |

---

## 2. Prérequis et installation

### 2.1 Ce dont vous avez besoin

Avant de lancer l'application, assurez-vous d'avoir :

- **Python 3.9 ou plus** installé sur votre machine
- Les **fichiers de données** Walmart (disponibles sur Kaggle)
- Les **artefacts du modèle** générés par le notebook d'entraînement

### 2.2 Structure des fichiers attendue

Votre dossier de projet doit ressembler à ceci :

```
walmart-sales-forecast/
│
├── app.py                        ← L'application (fichier principal)
│
├── data/                         ← Données brutes
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   └── features.csv
│
├── model_artifacts/              ← Modèle entraîné
│   ├── xgb_model.pkl
│   ├── features.pkl
│   ├── label_encoder_type.pkl
│   ├── target_encoding.pkl
│   ├── holidays.json
│   ├── historique_ventes.json
│   └── holdout_semaine.json      ← Requis pour le mode Hold-out
│
└── requirements.txt
```

>  **Important :** Si le dossier `model_artifacts/` est vide ou absent,
> vous devez d'abord exécuter entièrement le notebook `ProjetDataScienceModel.ipynb`.

### 2.3 Installation des dépendances

Ouvrez un terminal dans votre dossier projet et exécutez :

```bash
pip install -r requirements.txt
```

Si vous n'avez pas de fichier `requirements.txt`, installez manuellement :

```bash
pip install streamlit pandas numpy xgboost scikit-learn
```

---

## 3. Lancement de l'application

### Étape 1 — Ouvrir un terminal

Dans **VS Code** : `Ctrl + `` ` (backtick) ou **Terminal → New Terminal**

### Étape 2 — Se placer dans le bon dossier

```bash
cd chemin/vers/votre/projet
```

*Exemple sur Windows :*
```bash
cd C:/Users/VotreNom/Documents/WalmartProject
```

### Étape 3 — Lancer l'application

```bash
streamlit run app.py
```

### Étape 4 — Ouvrir dans le navigateur

Après quelques secondes, vous verrez dans le terminal :

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Cliquez sur le lien ou ouvrez `http://localhost:8501` dans votre navigateur.

>  **Astuce :** VS Code affiche souvent un bouton **"Open in Browser"** directement
> dans le terminal — cliquez dessus pour ouvrir l'app automatiquement.

---

## 4. Interface — Vue d'ensemble

L'interface se compose de **deux zones principales** :

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   SIDEBAR (gauche)          ZONE PRINCIPALE (droite)        │
│   ─────────────────         ──────────────────────────      │
│   ⚙️ Paramètres              📊 Résultats & Graphiques       │
│   • Choix du mode            • KPIs                         │
│   • Sélection boutique       • Tableau détaillé             │
│   • Sélection depts          • Graphiques                   │
│   • Semaine (libre)          • Historique des lags          │
│   • Bouton lancer            • Métriques (hold-out)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Mode Prédiction Libre — Guide pas à pas

Ce mode vous permet de prédire les ventes pour **n'importe quelle semaine**
disponible dans le fichier test.

---

### Étape 1 — Sélectionner le mode

Dans la sidebar, choisissez :

```
Mode : Prédiction libre
```

---

### Étape 2 — Choisir une boutique

Vous avez deux options :

**Option A — Tirage aléatoire** *(recommandé pour la démonstration)*

1. Sélectionnez `Aléatoire`
2. Cliquez sur **Tirer une boutique au sort**
3. La boutique tirée s'affiche dans l'encadré bleu

**Option B — Sélection manuelle**

1. Sélectionnez `Manuelle`
2. Choisissez un numéro de boutique dans la liste déroulante (1 à 45)

---

### Étape 3 — Choisir les départements

De la même façon, deux options sont disponibles :

**Option A — Tirage aléatoire de 6 départements**

1. Sélectionnez `Aléatoire (6)`
2. Cliquez sur **Tirer 6 départements**
3. Les départements sélectionnés apparaissent dans l'encadré vert

**Option B — Sélection manuelle**

1. Sélectionnez `Manuelle`
2. Choisissez jusqu'à **6 départements** dans la liste déroulante
   (maintenez `Ctrl` ou `Cmd` pour une sélection multiple)

---

### Étape 4 — Choisir la semaine

Utilisez le sélecteur de date pour choisir la semaine à prédire.

> 📅 La date par défaut est la **dernière semaine disponible** dans le fichier test.
> Vous pouvez remonter jusqu'au début du fichier test ou aller jusqu'à
> 4 semaines au-delà.

---

### Étape 5 — Lancer la prédiction

Cliquez sur le bouton **Lancer la Prédiction**

L'application calcule les prédictions et affiche les résultats en quelques secondes.

---

### Résultats obtenus

Vous obtenez :

- **4 indicateurs clés (KPIs)** : ventes totales prédites, nombre de départements
  analysés, meilleur et moins bon département
- **Un tableau détaillé** avec pour chaque département : les ventes prédites,
  la valeur du Lag_1 (ventes semaine précédente) et la moyenne glissante sur 4 semaines
- **Un graphique en barres** comparant les ventes prédites par département
- **Un graphique historique** montrant l'évolution des ventes réelles sur les
  2 derniers mois — ce sont exactement les données utilisées pour calculer les lags
- **Un indicateur de fête** si la semaine sélectionnée correspond à une fête
  américaine (Thanksgiving, Noël, Super Bowl, Labor Day)

---

## 6. Mode Évaluation Hold-out — Guide pas à pas

Ce mode est conçu pour **évaluer objectivement** la performance du modèle.
La semaine du **2012-10-26** a été isolée **avant** l'entraînement du modèle :
il ne l'a jamais vue, ce qui garantit une évaluation honnête.

---

### Étape 1 — Sélectionner le mode

Dans la sidebar, choisissez :

```
Mode :  Évaluation Hold-out (2012-10-26)
```

> La date est automatiquement verrouillée sur le **2012-10-26**.
> Vous ne pouvez pas la modifier — c'est voulu.

---

### Étape 2 — Choisir une boutique

Même procédure qu'en mode libre (aléatoire ou manuelle).

---

### Étape 3 — Choisir les départements

Même procédure qu'en mode libre.

>  **Conseil :** En mode Hold-out, seuls les départements ayant des données
> réelles pour cette boutique à cette date sont proposés, afin que la
> comparaison prédit vs réel soit possible.

---

### Étape 4 — Lancer la comparaison

Cliquez sur le bouton **Comparer Prédit vs Réel**

---

### Résultats obtenus

En plus des éléments du mode libre, vous obtenez :

**Un tableau de comparaison détaillé :**

| Département | Ventes Prédites ($) | Ventes Réelles ($) | Erreur ($) | Erreur (%) | Statut |
|---|---|---|---|---|---|
| 7 | $45,230 | $43,100 | +$2,130 | +4.9% | ✅ Bon |
| 14 | $12,400 | $15,600 | -$3,200 | -20.5% | ⚠️ Acceptable |

**Un graphique barres côte à côte** prédit vs réel par département

**4 métriques de performance globales :**

| Métrique | Description | Interprétation |
|---|---|---|
| **MAE** | Erreur absolue moyenne en $ | Plus c'est bas, mieux c'est |
| **RMSE** | Racine de l'erreur quadratique | Pénalise les grandes erreurs |
| **R²** | Coefficient de détermination | Proche de 1 = excellent |

**Une jauge de qualité automatique :**

```
 Excellente précision  →  MAE ≤ 10%
 Précision acceptable  →  MAE entre 10% et 20%
 Précision insuffisante →  MAE > 20%
```

---

## 7. Comprendre les résultats

### 7.1 Qu'est-ce que le Lag_1 ?

Le **Lag_1** représente les ventes réelles de la semaine précédente pour
ce même département dans cette même boutique. C'est la variable la plus
importante du modèle : si les ventes de la semaine passée étaient élevées,
le modèle prédit généralement des ventes élevées pour la semaine suivante.

### 7.2 Qu'est-ce que la Rolling_mean_4 ?

La **Rolling_mean_4** est la moyenne des ventes sur les 4 dernières semaines.
Elle lisse les anomalies ponctuelles et capture la tendance récente du département.

### 7.3 Pourquoi l'historique est limité à 2 mois ?

Pour des raisons de performance et de pertinence, seules les données des
**2 derniers mois** sont conservées en mémoire. Cela couvre largement
la fenêtre nécessaire pour calculer les lags (max. 4 semaines) et les
moyennes glissantes (max. 8 semaines).

### 7.4 Pourquoi certains départements manquent-ils ?

Certains départements n'existent pas dans toutes les boutiques. Si vous
ne trouvez pas un département précis, c'est simplement qu'il n'est pas
référencé pour la boutique sélectionnée.

### 7.5 Que signifient les statuts dans le tableau Hold-out ?

| Statut | Signification | Seuil |
|---|---|---|
| ✅ **Bon** | Prédiction très proche du réel | Erreur ≤ 10% |
| ⚠️ **Acceptable** | Légère déviation, mais utilisable | Erreur entre 10% et 20% |
| ❌ **Écart élevé** | Prédiction à interpréter avec prudence | Erreur > 20% |

---

## 8. Questions fréquentes (FAQ)

**L'application ne s'ouvre pas dans le navigateur**

Vérifiez que le terminal affiche bien `http://localhost:8501` et ouvrez
cette adresse manuellement dans votre navigateur. Si le port est occupé,
Streamlit utilise automatiquement le port suivant (`8502`, `8503`…).

---

**J'obtiens l'erreur "Fichier manquant"**

Cela signifie qu'un fichier dans `model_artifacts/` ou `data/` est absent.
Vérifiez que vous avez bien exécuté entièrement le notebook d'entraînement,
et que les fichiers CSV sont bien dans le dossier `data/`.

---

**Le mode Hold-out affiche un avertissement sur `holdout_semaine.json`**

Ce fichier est généré en fin de notebook. Ajoutez ce bloc dans la fonction
`sauvegarde_complete()` de votre notebook, avant la sauvegarde finale :

```python
holdout_save = holdout[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
holdout_save["Date"] = holdout_save["Date"].astype(str)
holdout_save.to_json(
    "./model_artifacts/holdout_semaine.json",
    orient="records",
    indent=2
)
```

---

**Les prédictions semblent très différentes des ventes réelles**

Plusieurs facteurs peuvent expliquer cela. Assurez-vous que la boutique
sélectionnée possède bien un historique de ventes dans les 2 derniers mois
du fichier d'entraînement — sans cela, les lags sont estimés par des
moyennes globales, ce qui réduit la précision.

---

**Puis-je prédire au-delà du fichier test ?**

Oui, jusqu'à 4 semaines au-delà de la dernière date du fichier test.
Au-delà, les données macroéconomiques (température, CPI, etc.) ne sont
plus disponibles et l'application utilise les dernières valeurs connues.

---

**Pour arrêter l'application**

Dans le terminal : `Ctrl + C`

---

## 9. Problèmes connus et solutions

| Problème | Cause probable | Solution |
|---|---|---|
| Page blanche au chargement | Chargement lent du modèle | Patienter 5-10 secondes au premier lancement |
| Boutique sans département | Boutique non présente dans `test.csv` | Choisir une autre boutique |
| Erreur `KeyError` sur une feature | Features du modèle incompatibles | Régénérer `features.pkl` depuis le notebook |
| MAPE à 0% en hold-out | Données réelles identiques aux prédites | Vérifier l'intégrité de `holdout_semaine.json` |
| Graphique historique vide | Aucune donnée dans la fenêtre des 2 mois | Normal pour certains couples boutique/département |

---

*Guide rédigé dans le cadre du projet Data Science — Walmart Sales Forecasting*
*Modèle : XGBoost optimisé · Interface : Streamlit · Données : Kaggle Walmart Dataset*
