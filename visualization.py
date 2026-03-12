# ============================================================
# visualization.py
# GROUPE G14 : D07 Emotion Detection + DistilBERT
# Section 8.2 du PDF : Résultats - Optimisation
#   - Tableau comparatif Baseline vs Optuna GridSampler
#   - Courbes de convergence
#   - Heatmap 12 combinaisons §4.2
#   - Historique des trials Optuna GridSampler
#   - Matrice de confusion
#
# ✅ Conforme optimization.py (GridSampler §7.1)
#    run_visualization(... grid_results, ...) — sans study ni grid_results_post
# Usage : from visualization import run_visualization
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import optuna


# ══════════════════════════════════════════════════════════
# 1. TABLEAU COMPARATIF — Baseline vs Optuna
# ══════════════════════════════════════════════════════════

def plot_comparison_table(baseline_results, best_results,
                           save_path='comparison_table.png'):
    """
    Tableau comparatif Baseline vs meilleur modèle Optuna GridSampler.
    Section 8.2 du PDF — Grid Search exhaustif §4.2 via Optuna GridSampler §7.1.
    """
    splits = baseline_results["split"]

    rows = []
    for i, split in enumerate(splits):
        acc_base = baseline_results["accuracy"][i]
        acc_best = best_results["accuracy"][i]
        f1_base  = baseline_results["f1_macro"][i]
        f1_best  = best_results["f1_macro"][i]
        gain_acc = (acc_best - acc_base) * 100
        gain_f1  = (f1_best  - f1_base)  * 100
        rows.append({
            "Split"              : split,
            "Accuracy (Baseline)": f"{acc_base:.4f}",
            "Accuracy (Grid Search)": f"{acc_best:.4f}",
            "F1 (Baseline)"      : f"{f1_base:.4f}",
            "F1 (Grid Search)"   : f"{f1_best:.4f}",
            "Gain Accuracy"      : f"{gain_acc:+.2f}%",
            "Gain F1"            : f"{gain_f1:+.2f}%",
        })

    df = pd.DataFrame(rows)

    # Affichage console
    print("\n" + "="*75)
    print("📋 TABLEAU COMPARATIF — BASELINE vs GRID SEARCH (section 8.2)")
    print("="*75)
    print(df.to_string(index=False))

    # Figure matplotlib
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    table = ax.table(
        cellText    = df.values,
        colLabels   = df.columns,
        cellLoc     = 'center',
        loc         = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Colorier les gains positifs en vert, négatifs en rouge
    for i in range(len(rows)):
        for j, col in enumerate(df.columns):
            if col in ("Gain Accuracy", "Gain F1"):
                val = df.iloc[i][col]
                color = '#d4edda' if '+' in val else '#f8d7da'
                table[i+1, j].set_facecolor(color)

    # Header en bleu
    for j in range(len(df.columns)):
        table[0, j].set_facecolor('#cce5ff')
        table[0, j].set_text_props(weight='bold')

    plt.title('Comparaison Baseline vs Optuna GridSampler P02 (section 8.2, §4.2)',
              fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Tableau sauvegardé : {save_path}")

    return df


# ══════════════════════════════════════════════════════════
# 2. COURBES DE CONVERGENCE
# ══════════════════════════════════════════════════════════

def plot_convergence_curves(baseline_trainer, best_trainer,
                             save_path='convergence_curves.png'):
    """
    Courbes de convergence Baseline vs meilleur modèle.
    Train loss + Val loss + Val F1-macro.
    Section 8.2 du PDF.
    """
    # Extraire logs
    def extract_logs(trainer):
        logs      = trainer.state.log_history
        train_loss = [(x['step'], x['loss'])
                      for x in logs if 'loss' in x and 'eval_loss' not in x]
        eval_logs  = [(x['epoch'], x['eval_loss'], x['eval_f1_macro'])
                      for x in logs
                      if 'eval_loss' in x and 'eval_f1_macro' in x]
        return train_loss, eval_logs

    tl_base, el_base = extract_logs(baseline_trainer)
    tl_best, el_best = extract_logs(best_trainer)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Courbes de Convergence — Baseline vs Grid Search (section 8.2)',
                 fontsize=13, fontweight='bold')

    # ── Graphe 1 : Train Loss ────────────────────────────
    if tl_base:
        steps_b, loss_b = zip(*tl_base)
        axes[0].plot(steps_b, loss_b, 'r-o', markersize=4, label='Baseline')
    if tl_best:
        steps_g, loss_g = zip(*tl_best)
        axes[0].plot(steps_g, loss_g, 'b-s', markersize=4, label='Grid Search')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Graphe 2 : Val Loss ──────────────────────────────
    if el_base:
        ep_b, vl_b, _ = zip(*el_base)
        axes[1].plot(ep_b, vl_b, 'r-o', markersize=5, label='Baseline')
    if el_best:
        ep_g, vl_g, _ = zip(*el_best)
        axes[1].plot(ep_g, vl_g, 'b-s', markersize=5, label='Grid Search')
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoque')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Graphe 3 : Val F1-macro ──────────────────────────
    if el_base:
        ep_b, _, f1_b = zip(*el_base)
        axes[2].plot(ep_b, f1_b, 'r-o', markersize=5, label='Baseline')
    if el_best:
        ep_g, _, f1_g = zip(*el_best)
        axes[2].plot(ep_g, f1_g, 'b-s', markersize=5, label='Grid Search')
    axes[2].set_title('Validation F1-macro')
    axes[2].set_xlabel('Epoque')
    axes[2].set_ylabel('F1-macro')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Courbes sauvegardées : {save_path}")


# ══════════════════════════════════════════════════════════
# 3. HEATMAP COMPLÈTE — Post-Optuna Grid Evaluation
# ══════════════════════════════════════════════════════════

def plot_heatmap_complete_grid(grid_results, save_path='heatmap_complete_grid.png'):
    """
    Heatmap complète des 12 combinaisons weight_decay x dropout.
    Grid Search exhaustif Optuna §4.2 — 12 combinaisons weight_decay × dropout.
    Section 4.2 du PDF — Hyperparamètres fixes (4 × 3 = 12 combinaisons).
    """
    df    = pd.DataFrame(grid_results)
    pivot = df.pivot(
        index   = 'dropout',
        columns = 'weight_decay',
        values  = 'val_accuracy'
    )
    pivot_f1 = df.pivot(
        index   = 'dropout',
        columns = 'weight_decay',
        values  = 'val_f1_macro'
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Heatmap Grid Search Optuna — weight_decay × dropout (section §4.2 + §7.1)',
                 fontsize=13, fontweight='bold')

    # Heatmap Accuracy
    sns.heatmap(
        pivot,
        annot      = True,
        fmt        = '.3f',
        cmap       = 'RdYlGn',
        linewidths = 0.8,
        ax         = axes[0],
        vmin       = 0.0,
        vmax       = 1.0,
        annot_kws  = {"size": 11}
    )
    axes[0].set_title('Accuracy Validation\n(12 combinaisons : 4 weight_decay × 3 dropout)')
    axes[0].set_xlabel('Weight Decay (échelle log)')
    axes[0].set_ylabel('Dropout')
    col_labels = [f'{v:.0e}' for v in sorted(df['weight_decay'].unique())]
    axes[0].set_xticklabels(col_labels, rotation=45)

    # Heatmap F1-macro
    sns.heatmap(
        pivot_f1,
        annot      = True,
        fmt        = '.3f',
        cmap       = 'RdYlGn',
        linewidths = 0.8,
        ax         = axes[1],
        vmin       = 0.0,
        vmax       = 1.0,
        annot_kws  = {"size": 11}
    )
    axes[1].set_title('F1-macro Validation\n(12 combinaisons : 4 weight_decay × 3 dropout)')
    axes[1].set_xlabel('Weight Decay (échelle log)')
    axes[1].set_ylabel('Dropout')
    axes[1].set_xticklabels(col_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Heatmap complète (12 combinaisons) sauvegardée : {save_path}")


# ══════════════════════════════════════════════════════════
# 4. RÉSUMÉ DES HYPERPARAMÈTRES — Optuna GridSampler
# ══════════════════════════════════════════════════════════

def plot_optuna_importance(grid_results, save_path='optuna_importance.png'):
    """
    Visualise l'accuracy validation de chaque combinaison weight_decay × dropout.
    Version compatible GridSampler — affiche accuracy val par combinaison.
    Section 8.2 du PDF — Résultats Optuna GridSampler §7.1.
    """
    # GridSampler : afficher accuracy val par combinaison (weight_decay fixé, dropout en couleur)
    df  = pd.DataFrame(grid_results)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors_map = {0.0: '#4472C4', 0.1: '#FF6B6B', 0.3: '#70AD47'}
    for do in sorted(df['dropout'].unique()):
        sub = df[df['dropout'] == do].sort_values('weight_decay')
        ax.plot(
            sub['weight_decay'].astype(str),
            sub['val_accuracy'],
            marker='o', linewidth=2, markersize=8,
            label=f'dropout={do}',
            color=colors_map.get(do, 'gray')
        )

    ax.set_xlabel('Weight Decay', fontsize=11)
    ax.set_ylabel('Accuracy validation', fontsize=11)
    ax.set_title(
        'Effet weight_decay × dropout sur la généralisation\n'
        'Optuna GridSampler §4.2 + §7.1 (section §8.2)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Graphe hyperparamètres sauvegardé : {save_path}")


# ══════════════════════════════════════════════════════════
# 4. HISTORIQUE DES TRIALS — Optuna convergence
# ══════════════════════════════════════════════════════════

def plot_optuna_history(grid_results, save_path='optuna_history.png'):
    """
    Historique des 12 trials Optuna GridSampler — accuracy val par trial.
    Les trials élagués (TrialPruned §5.2) sont exclus de grid_results par optimization.py.
    Section 8.2 du PDF — Résultats Optuna GridSampler §7.1.
    """
    # Basé sur grid_results (liste de dicts) — compatible GridSampler §7.1
    df         = pd.DataFrame(grid_results)
    n_trials   = len(df)
    trial_nums = list(range(1, n_trials + 1))
    accuracies = df['val_accuracy'].tolist()

    # Meilleure accuracy cumulée
    best_vals = []
    best_so_far = -np.inf
    for v in accuracies:
        best_so_far = max(best_so_far, v)
        best_vals.append(best_so_far)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Historique Optuna GridSampler — 12 trials §4.2 (section §8.2, §7.1)',
                 fontsize=12, fontweight='bold')

    # Graphe 1 : accuracy val par trial + best so far
    axes[0].plot(trial_nums, best_vals, 'b-o', linewidth=2.5, markersize=6, label='Best so far')
    axes[0].scatter(trial_nums, accuracies, color='green', s=80,
                    marker='o', label='COMPLETE', zorder=5)
    axes[0].set_xlabel('Numéro du trial', fontsize=11)
    axes[0].set_ylabel('Accuracy validation', fontsize=11)
    axes[0].set_title('Convergence : Meilleure Accuracy par Trial\n(GridSampler §7.1 — 12 combinaisons §4.2)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Graphe 2 : accuracy par combinaison dropout (barplot groupé)
    dropout_vals = sorted(df['dropout'].unique())
    colors_map   = {0.0: '#4472C4', 0.1: '#FF6B6B', 0.3: '#70AD47'}
    x    = np.arange(len(df['weight_decay'].unique()))
    wds  = sorted(df['weight_decay'].unique())
    width = 0.25

    for i, do in enumerate(dropout_vals):
        sub  = df[df['dropout'] == do].sort_values('weight_decay')
        vals = sub['val_accuracy'].tolist()
        axes[1].bar(x + i*width, vals, width,
                    label=f'dropout={do}',
                    color=colors_map.get(do, 'gray'),
                    edgecolor='black', alpha=0.85)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f'{v:.0e}' for v in wds], rotation=30)
    axes[1].set_xlabel('Weight Decay', fontsize=11)
    axes[1].set_ylabel('Accuracy validation', fontsize=11)
    axes[1].set_title('Accuracy par combinaison (§4.2)\n4 weight_decay × 3 dropout')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Historique Optuna sauvegardé : {save_path}")

    # Résumé console
    print("\n" + "="*60)
    print(f"📊 Résumé Optuna GridSampler :")
    print(f"   Total trials  : {n_trials}")
    print(f"   COMPLETE      : {n_trials}  (tous testés — GridSampler exhaustif)")
    print(f"   Meilleure accuracy : {best_so_far:.4f}")
    print("="*60)


# ══════════════════════════════════════════════════════════
# 5. MATRICE DE CONFUSION
# ══════════════════════════════════════════════════════════

def plot_confusion_matrix(best_trainer, test_data, label_names,
                           save_path='confusion_matrix.png'):
    """
    Matrice de confusion du meilleur modèle sur le test set.
    Section 8.2 du PDF.
    """
    preds       = best_trainer.predict(test_data)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = preds.label_ids

    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Matrice de Confusion — Meilleur modèle Grid Search (section 8.2)',
                 fontsize=12, fontweight='bold')

    # Matrice absolue
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title('Valeurs absolues')
    axes[0].set_xlabel('Prédit')
    axes[0].set_ylabel('Réel')
    axes[0].tick_params(axis='x', rotation=45)

    # Matrice normalisée
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_title('Valeurs normalisées (recall par classe)')
    axes[1].set_xlabel('Prédit')
    axes[1].set_ylabel('Réel')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Matrice de confusion sauvegardée : {save_path}")

    # Rapport détaillé
    print("\n📋 Rapport de classification détaillé :")
    print(classification_report(true_labels, pred_labels,
                                 target_names=label_names))


# ══════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ══════════════════════════════════════════════════════════

def run_visualization(baseline_results, best_results,
                       baseline_trainer, best_trainer,
                       grid_results, test_data, label_names):
    """
    Génère toutes les visualisations section 8.2 du PDF.
    Grid Search Optuna GridSampler §7.1 — 12 combinaisons §4.2.

    Paramètres
    ----------
    baseline_results : dict — résultats baseline (train/val/test)
    best_results     : dict — résultats meilleur modèle Optuna
    baseline_trainer : Trainer — trainer baseline (pour courbes)
    best_trainer     : Trainer — trainer meilleur modèle
    grid_results     : list — résultats des 12 combinaisons §4.2 (de run_gridsearch)
    test_data        : dataset tokenisé test
    label_names      : list — noms des classes

    Fichiers générés
    ----------------
    comparison_table.png       — tableau comparatif baseline vs Optuna
    convergence_curves.png     — courbes train loss + val loss + val F1
    heatmap_complete_grid.png  — heatmap 12 combinaisons (4 × 3)
    optuna_importance.png      — graphe weight_decay × dropout (GridSampler)
    optuna_history.png         — historique des 12 trials GridSampler
    confusion_matrix.png       — matrice de confusion meilleur modèle
    """
    print("\n📊 Génération des visualisations (section 8.2 — Optuna GridSampler §7.1)...\n")

    # 1 — Tableau comparatif
    print("📋 Tableau comparatif Baseline vs Optuna...")
    df_comparison = plot_comparison_table(baseline_results, best_results)

    # 2 — Courbes de convergence
    print("\n📈 Courbes de convergence...")
    plot_convergence_curves(baseline_trainer, best_trainer)

    # 3 — Heatmap complète (12 combinaisons post-Optuna)
    print("\n🗺️  Heatmap complète (4 weight_decay × 3 dropout)...")
    plot_heatmap_complete_grid(grid_results)

    # 4 — Importance des hyperparamètres (Optuna)
    print("\n🎯 Graphe hyperparamètres (Optuna GridSampler)...")
    plot_optuna_importance(grid_results)

    # 5 — Historique des trials (Optuna)
    print("\n📉 Historique Optuna GridSampler (12 trials)...")
    plot_optuna_history(grid_results)

    # 6 — Matrice de confusion
    print("\n🔲 Matrice de confusion...")
    plot_confusion_matrix(best_trainer, test_data, label_names)

    print("\n" + "="*70)
    print("✅ TOUTES LES VISUALISATIONS GÉNÉRÉES (section 8.2)")
    print("   comparison_table.png")
    print("   convergence_curves.png")
    print("   heatmap_complete_grid.png     [Évaluation POST-OPTUNA : 12 combinaisons]")
    print("   optuna_importance.png         [Optuna GridSampler — §4.2 × §7.1]")
    print("   optuna_history.png            [Optuna GridSampler — 12 trials §7.1]")
    print("   confusion_matrix.png")
    print("="*70)

    return df_comparison
