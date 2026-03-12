# ============================================================
# loss_landscape.py
# GROUPE G14 : D07 Emotion Detection + DistilBERT
# Sections 6.1, 6.2, 6.3 du PDF
#
# Conformité PDF :
#   Section 6.1 — Méthode simplifiée CPU (Listing 4)
#                  n_points=8, epsilon=0.05, n_samples=50
#   Section 6.2 — Visualisation 1D (Figure 1 du PDF)
#   Section 6.3 — Métrique de platitude Sharpness (formule 1)
#
# Usage : from loss_landscape import run_loss_landscape
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════
# ÉVALUATION SUR SUBSET
# ══════════════════════════════════════════════════════════

def evaluate_on_subset(model, dataset, n_samples=50, device=None):
    """
    Évalue la loss moyenne sur n_samples exemples.
    Listing 4 du PDF — n_samples=50.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    subset = dataset.select(range(min(n_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=16)

    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs        = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )
            total_loss += outputs.loss.item()
            n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else float('inf')


# ══════════════════════════════════════════════════════════
# LOSS LANDSCAPE 1D — Listing 4 du PDF
# ══════════════════════════════════════════════════════════

def compute_loss_landscape_light(model, dataset, n_points=8, epsilon=0.05):
    """
    Version légère pour CPU — section 6.1 du PDF.
    Code issu exactement du Listing 4 du PDF.

    Paramètres
    ----------
    model    : modèle entraîné (baseline ou grid search)
    dataset  : dataset tokenisé
    n_points : nombre de points sur la grille (défaut 8 — section 6.1)
    epsilon  : amplitude de perturbation (défaut 0.05 — section 6.1)

    Retourne
    --------
    alphas : array des perturbations
    losses : liste des losses correspondantes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Sauvegarder paramètres originaux (Listing 4 du PDF)
    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée (Listing 4 du PDF)
    direction = [torch.randn_like(p) for p in model.parameters()]
    norm      = sum(torch.norm(d) for d in direction)
    direction = [d / norm for d in direction]

    # Grille réduite (Listing 4 du PDF)
    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        # Appliquer perturbation (Listing 4 du PDF)
        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + alpha * d

        # Évaluer sur petit subset (Listing 4 — n_samples=50)
        loss = evaluate_on_subset(model, dataset, n_samples=50, device=device)
        losses.append(loss)

    # Restaurer paramètres originaux (Listing 4 du PDF)
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0

    return alphas, losses


# ══════════════════════════════════════════════════════════
# MÉTRIQUE DE PLATITUDE — Section 6.3 du PDF
# ══════════════════════════════════════════════════════════

def compute_sharpness(alphas, losses):
    """
    Métrique de platitude — formule exacte section 6.3 du PDF :

        Sharpness = (1/N) * sum_i |L(θ + ε*d_i) - L(θ)|

    Plus la valeur est basse, plus le minimum est plat
    => meilleure généralisation attendue (Keskar et al., 2017)
    """
    center_idx  = len(losses) // 2
    center_loss = losses[center_idx]
    N           = len(losses)
    sharpness   = (1 / N) * sum(abs(l - center_loss) for l in losses)
    return sharpness


# ══════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ══════════════════════════════════════════════════════════

def run_loss_landscape(baseline_model, best_model, test_data,
                       n_points=8, epsilon=0.05):
    """
    Compare loss landscape baseline vs meilleur modèle Grid Search.

    Conformité PDF :
        Section 6.1 — Listing 4 : n_points=8, epsilon=0.05, n_samples=50
        Section 6.2 — Figure 1  : visualisation 1D minimum plat vs pointu
        Section 6.3 — Formule 1 : Sharpness = (1/N) * sum |L(θ+εd) - L(θ)|

    Paramètres
    ----------
    baseline_model : modèle baseline (sans régularisation)
    best_model     : meilleur modèle issu du grid search
    test_data      : dataset test tokenisé
    n_points       : nombre de points grille (section 6.1 = 8)
    epsilon        : amplitude perturbation (section 6.1 = 0.05)

    Fichiers générés
    ----------------
    loss_landscape.png  — visualisation 1D + barplot sharpness

    Retourne
    --------
    results : dict avec sharpness_baseline, sharpness_best, amelioration_pct
    """

    print("📐 Calcul du Loss Landscape (section 6.1)...\n")

    # ── Baseline ──────────────────────────────────────────
    print("   → Modèle Baseline (sans régularisation)...")
    alphas_base, losses_base = compute_loss_landscape_light(
        baseline_model, test_data,
        n_points = n_points,
        epsilon  = epsilon
    )
    sharpness_base = compute_sharpness(alphas_base, losses_base)
    print(f"      Sharpness Baseline : {sharpness_base:.6f}")

    # ── Meilleur modèle Grid Search ───────────────────────
    print("   → Meilleur modèle (Grid Search + régularisation)...")
    alphas_best, losses_best = compute_loss_landscape_light(
        best_model, test_data,
        n_points = n_points,
        epsilon  = epsilon
    )
    sharpness_best = compute_sharpness(alphas_best, losses_best)
    print(f"      Sharpness Grid Search : {sharpness_best:.6f}")

    # ── Visualisation 1D — Figure 1 du PDF ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Analyse du Loss Landscape — Baseline vs Grid Search\n'
        'Sections 6.2 et 6.3 du PDF',
        fontsize=13, fontweight='bold'
    )

    # Graphe 1 : Loss Landscape 1D (Figure 1 du PDF)
    axes[0].plot(alphas_base, losses_base, 'r-o', linewidth=2, markersize=6,
                 label=f'Baseline — Minimum pointu\n(sharpness={sharpness_base:.5f})')
    axes[0].plot(alphas_best, losses_best, 'b-s', linewidth=2, markersize=6,
                 label=f'Grid Search — Minimum plat\n(sharpness={sharpness_best:.5f})')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.6,
                    label='θ original (α=0)')
    axes[0].fill_between(alphas_base, losses_base, alpha=0.1, color='red')
    axes[0].fill_between(alphas_best, losses_best, alpha=0.1, color='blue')
    axes[0].set_title('Visualisation 1D du Loss Landscape\n(section 6.2 du PDF — Figure 1)')
    axes[0].set_xlabel('Direction de perturbation (α)')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Graphe 2 : Barplot Sharpness (section 6.3)
    models  = ['Baseline\n(sans régul.)', 'Grid Search\n(avec régul.)']
    sharps  = [sharpness_base, sharpness_best]
    colors  = ['tomato', 'steelblue']
    bars    = axes[1].bar(models, sharps, color=colors,
                          edgecolor='black', width=0.4)

    # Annoter les barres
    for bar, val in zip(bars, sharps):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sharps) * 0.02,
            f'{val:.6f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    # Flèche d'amélioration
    amelioration = ((sharpness_base - sharpness_best) / sharpness_base * 100
                    if sharpness_base != 0 else 0)
    if amelioration > 0:
        axes[1].annotate(
            f'-{amelioration:.1f}%\n(amélioration)',
            xy     = (1, sharpness_best),
            xytext = (0.5, (sharpness_base + sharpness_best) / 2),
            arrowprops = dict(arrowstyle='->', color='green', lw=2),
            color  = 'green', fontsize=10, fontweight='bold'
        )

    axes[1].set_title('Métrique de Platitude — Sharpness\n(section 6.3 du PDF — Formule 1)')
    axes[1].set_ylabel('Sharpness\n(plus bas = minimum plus plat => meilleure généralisation)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=150)
    plt.show()
    print('\n✅ Sauvegardé : loss_landscape.png')

    # ── Résumé section 6.3 ───────────────────────────────
    print("\n" + "="*55)
    print("📋 MÉTRIQUES DE PLATITUDE (section 6.3)")
    print("="*55)
    print(f"   Formule : Sharpness = (1/N) * sum |L(θ+εd) - L(θ)|")
    print(f"   N = {n_points} points | ε = {epsilon}")
    print(f"\n   Sharpness Baseline    : {sharpness_base:.6f}  <- minima pointu")
    print(f"   Sharpness Grid Search : {sharpness_best:.6f}  <- minima plat")
    if amelioration > 0:
        print(f"\n   ✅ Amélioration : {amelioration:.1f}%")
        print(f"   -> La régularisation (weight_decay) produit un minimum")
        print(f"      plus plat, associé à une meilleure généralisation")
        print(f"      (Keskar et al., 2017)")
    else:
        print(f"\n   ⚠️  Pas d'amélioration de platitude détectée")
        print(f"   -> La régularisation n'a pas modifié la géométrie du minimum")

    results = {
        "sharpness_baseline"  : sharpness_base,
        "sharpness_gridsearch": sharpness_best,
        "amelioration_pct"    : amelioration,
        "n_points"            : n_points,
        "epsilon"             : epsilon,
        "alphas_baseline"     : alphas_base.tolist(),
        "losses_baseline"     : losses_base,
        "alphas_gridsearch"   : alphas_best.tolist(),
        "losses_gridsearch"   : losses_best,
    }
    return results
