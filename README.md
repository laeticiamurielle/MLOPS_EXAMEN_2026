# Projet MLOPS 2026

## Groupe G14 — Fine-tuning DistilBERT | Régularisation & Généralisation

| Paramètre | Valeur |
|---|---|
| **Dataset** | D07 — Emotion Detection (6 classes, 20k exemples) |
| **Modèle** | M01 — DistilBERT-base-uncased (66M paramètres) |
| **Problématique** | P02 — Comment le weight decay et le dropout affectent-ils la généralisation ? |
| **Méthode d'optimisation** | Optuna (recherche Bayésienne) |
| **Métrique principale** | F1-score macro |
| **Date limite** | 13 mars 2026 |

---

## Structure du projet

```
projet_G14/
├── requirements.txt          ← dépendances Python
├── notebook_final.ipynb      ← notebook principal 
└── src/
    ├── __init__.py
    ├── data_loader.py        ← Étape 1 : chargement et sous-échantillonnage 
    ├── model_setup.py        ← Étape 1 : initialisation adaptative du modèle 
    ├── baseline.py           ← Étape 2 : entraînement baseline sans régularisation 
    ├── optimization.py       ← Étape 3 : grid search Optuna weight_decay × dropout 
    ├── loss_landscape.py     ← Étape 4 : analyse du loss landscape 1D 
    └── visualization.py      ← Étape 5 : visualisations et comparaison finale 
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <https://github.com/laeticiamurielle/MLOPS_EXAMEN_2026.git>
cd projet_G14
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv .mlopsenv
# Windows
.mlopsenv\Scripts\activate
# Linux / Mac
source .mlopsenv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Exécution

Ouvre `notebook_final.ipynb` dans VSCode et exécute les cellules dans l'ordre :

| Étape | Cellule | Description |
|---|---|---|
| 1 | `from data_loader import load_data_as_dataframe` | Chargement et tokenisation |
| 2 | `from baseline import run_baseline` | Entraînement baseline |
| 3 | `from optimization import run_optuna` | Optimisation Optuna |
| 4 | `from loss_landscape import run_loss_landscape` | Analyse loss landscape |
| 5 | `from visualization import run_visualization` | Visualisations finales |

---

## Protocole P02 — Régularisation et Généralisation

- **Grid search** sur `weight_decay` : `[1e-5, 1e-4, 1e-3, 1e-2]`
- **Grid search** sur `dropout` : `[0.0, 0.1, 0.3]`
- Mesure de l'**écart train/test** à chaque configuration
- Analyse de la **platitude des minima** (Sharpness — section 6.3)

---

## Adaptation CPU 

| Contrainte | Solution appliquée |
|---|---|
| Pas de GPU | `torch_dtype=float32`, `torch.set_num_threads(4)` |
| RAM limitée | `max_length=128`, batch size 8 ou 16 |
| Temps contraint | `max_steps=100` par trial Optuna, sous-ensembles de 500/100 |

---

## Fichiers générés

Après exécution complète du notebook :

```
distribution_classes.png      ← distribution des 6 émotions
baseline_curves.png           ← courbes loss/F1 baseline
optuna_history.png            ← évolution F1 sur les 20 trials
loss_landscape.png            ← comparaison platitude minima (section 6.2)
convergence_curves.png        ← baseline vs meilleur modèle
heatmap_regularization.png    ← weight_decay × dropout → accuracy
confusion_matrix.png          ← matrice de confusion finale
```

---

## Dépendances principales

```
torch==2.1.0
transformers==4.40.0
datasets==2.18.0
numpy==1.26.4
scikit-learn==1.3.2
optuna
accelerate
```

---

## Contact

**Enseignant :** mbialaura12@gmail.com  
**Rendu :** Par mail — Rapport PDF + lien GitHub
**Le navigateur s'ouvre automatiquement sur http://localhost:8501