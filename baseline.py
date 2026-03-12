# ============================================================
# baseline.py
# GROUPE G14 : D07 Emotion Detection + DistilBERT
# Étape 2 : Baseline sans régularisation 
# ✅ Système de cache intégré — évite de réentraîner à chaque fois
# Usage : from baseline import run_baseline
# ============================================================

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import (
    TrainingArguments, Trainer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model_setup import get_model_and_tokenizer

CACHE_DIR = "./baseline_cache"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}


def _save_cache(baseline_results, trainer, cache_dir):
    """Sauvegarde les résultats et le modèle dans le cache"""
    os.makedirs(cache_dir, exist_ok=True)

    # Résultats
    with open(f'{cache_dir}/baseline_results.pkl', 'wb') as f:
        pickle.dump(baseline_results, f)

    # Modèle
    trainer.save_model(f'{cache_dir}/baseline_model')

    # Historique des logs (courbes)
    with open(f'{cache_dir}/trainer_state.json', 'w') as f:
        json.dump({"log_history": trainer.state.log_history}, f)

    print(f"\n💾 Baseline sauvegardée dans '{cache_dir}/'")
    print("   → Prochain appel : chargement instantané depuis le cache ✅")


def _load_cache(cache_dir, label_names):
    """Charge la baseline depuis le cache"""
    print("\n⚡ Cache détecté — chargement instantané (pas de réentraînement)")

    # Résultats
    with open(f'{cache_dir}/baseline_results.pkl', 'rb') as f:
        baseline_results = pickle.load(f)

    # Modèle
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        f'{cache_dir}/baseline_model'
    )

    # Trainer minimal pour les courbes
    trainer = Trainer(
        model = baseline_model,
        args  = TrainingArguments(
            output_dir = cache_dir,
            report_to  = "none"
        ),
    )

    # Recharger l'historique des logs
    with open(f'{cache_dir}/trainer_state.json', 'r') as f:
        state_data = json.load(f)
    trainer.state.log_history = state_data.get('log_history', [])

    # Afficher résultats
    df = pd.DataFrame(baseline_results)
    print("\n📋 RÉSULTATS BASELINE (depuis cache)")
    print("="*50)
    print(df.to_string(index=False))
    gap = baseline_results["accuracy"][0] - baseline_results["accuracy"][2]
    print(f"\n⚠️  Écart Train-Test : {gap:.4f}")

    return baseline_results, trainer, baseline_model


def run_baseline(train_data, val_data, test_data, label_names,
                 output_dir="./baseline_model",
                 cache_dir=CACHE_DIR,
                 force_retrain=False):
    """
    Entraîne DistilBERT avec hyperparamètres par défaut.
    Sert de point de référence pour P02 (section 4.2).

    ✅ Cache automatique : si déjà entraîné, charge depuis le disque.
    Pour forcer un réentraînement : force_retrain=True

    Paramètres
    ----------
    force_retrain : bool — forcer le réentraînement même si cache existe

    Retourne
    --------
    baseline_results, trainer, model
    """

    # ── Vérifier si le cache existe ───────────────────────
    cache_exists = (
        os.path.exists(f'{cache_dir}/baseline_results.pkl') and
        os.path.exists(f'{cache_dir}/baseline_model') and
        os.path.exists(f'{cache_dir}/trainer_state.json')
    )

    if cache_exists and not force_retrain:
        return _load_cache(cache_dir, label_names)

    if force_retrain:
        print("🔄 Réentraînement forcé (force_retrain=True)")

    # ── Chargement modèle (section 3.3 du PDF) ────────────
    model, tokenizer, device = get_model_and_tokenizer(
        num_labels = len(label_names),
        dropout    = 0.1
    )

    # ── Hyperparamètres par défaut (baseline) ─────────────
    print("\n⚙️  Hyperparamètres Baseline (sans tuning) :")
    default_params = {
        "learning_rate"              : 5e-5,
        "num_train_epochs"           : 3,
        "per_device_train_batch_size": 16,
        "weight_decay"               : 0.0,
        "warmup_steps"               : 0,
        "dropout"                    : 0.1,
    }
    for k, v in default_params.items():
        print(f"   {k:<35} : {v}")

    # ── Trainer ───────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        learning_rate               = default_params["learning_rate"],
        num_train_epochs            = default_params["num_train_epochs"],
        per_device_train_batch_size = default_params["per_device_train_batch_size"],
        per_device_eval_batch_size  = default_params["per_device_train_batch_size"],
        weight_decay                = default_params["weight_decay"],
        warmup_steps                = default_params["warmup_steps"],
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        logging_steps               = 20,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_data,
        eval_dataset    = val_data,
        compute_metrics = compute_metrics,
    )

    # ── Entraînement ──────────────────────────────────────
    print("\n🚀 Entraînement Baseline (une seule fois)...")
    trainer.train()
    print("   ✅ Entraînement terminé !")

    # ── Évaluation ────────────────────────────────────────
    print("\n📊 Évaluation sur train / val / test...")
    train_eval = trainer.evaluate(train_data)
    val_eval   = trainer.evaluate(val_data)
    test_eval  = trainer.evaluate(test_data)

    baseline_results = {
        "split"    : ["Train",                     "Validation",              "Test"],
        "accuracy" : [train_eval["eval_accuracy"], val_eval["eval_accuracy"], test_eval["eval_accuracy"]],
        "f1_macro" : [train_eval["eval_f1_macro"], val_eval["eval_f1_macro"], test_eval["eval_f1_macro"]],
    }

    df = pd.DataFrame(baseline_results)
    print("\n" + "="*50)
    print("📋 RÉSULTATS BASELINE — point de référence P02")
    print("="*50)
    print(df.to_string(index=False))

    gap = baseline_results["accuracy"][0] - baseline_results["accuracy"][2]
    print(f"\n⚠️  Écart Train-Test : {gap:.4f}")

    preds       = trainer.predict(test_data)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    print("\n📋 Rapport détaillé (test) :")
    print(classification_report(preds.label_ids, pred_labels, target_names=label_names))

    # ── Sauvegarder dans le cache ─────────────────────────
    _save_cache(baseline_results, trainer, cache_dir)

    return baseline_results, trainer, model
