# ============================================================
# src/model_setup.py
# GROUPE G14 : D07 Emotion Detection + DistilBERT
# Respecte exactement la section 3.3 du PDF
# Usage : from src.model_setup import get_model_and_tokenizer
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"


def get_model_and_tokenizer(model_name=MODEL_NAME, num_labels=6,
                             device=None, dropout=0.1):
    """
    Charge modèle et tokenizer avec adaptation automatique.
    Code issu exactement de la section 3.3 du PDF.

    Paramètres
    ----------
    model_name : str   — nom HuggingFace du modèle
    num_labels : int   — nombre de classes (6 pour Emotion)
    device     : str   — 'cpu' ou 'cuda' (auto-détecté si None)
    dropout    : float — taux de dropout (pour P02 régularisation)

    Retourne
    --------
    model, tokenizer, device
    """

    # ── Détection automatique du device (section 3.3) ─────
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Utilisation de : {device}")

    # ── Chargement du tokenizer (section 3.3) ──────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Chargement du modèle avec optimisations CPU ────────
    # Section 3.3 : float32 sur CPU, float16 sur GPU
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels                   = num_labels,
        seq_classif_dropout          = dropout,
        torch_dtype = torch.float32 if device.type == 'cpu' else torch.float16
    )

    model = model.to(device)

    # ── Optimisations pour CPU (section 3.3 du PDF) ────────
    if device.type == 'cpu':
        torch.set_num_threads(4)   # Ajuster selon votre CPU
        print("   Optimisations CPU activées (4 threads)")

    return model, tokenizer, device
