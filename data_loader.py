# ============================================================
# data_loader.py
# GROUPE G14 : D07 Emotion Detection + DistilBERT
# ✅ Cache intégré — évite de retélécharger à chaque Restart
# Usage : from data_loader import load_data_as_dataframe
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
N_CLASSES  = 6
CACHE_DIR  = "./data_cache"


# ── Fonctions cache ────────────────────────────────────────

def _save_cache(train_data, val_data, test_data, tokenizer, label_names, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    with open(f'{cache_dir}/splits.pkl', 'wb') as f:
        pickle.dump({
            'train'      : train_data,
            'val'        : val_data,
            'test'       : test_data,
            'label_names': label_names,
        }, f)
    tokenizer.save_pretrained(f'{cache_dir}/tokenizer')
    print(f"💾 Données sauvegardées dans '{cache_dir}/'")
    print("   → Prochain appel : chargement instantané ✅")


def _load_cache(cache_dir):
    print("⚡ Cache détecté — chargement instantané (pas de retéléchargement)")
    with open(f'{cache_dir}/splits.pkl', 'rb') as f:
        data = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(f'{cache_dir}/tokenizer')
    return (
        data['train'], data['val'], data['test'],
        tokenizer, data['label_names']
    )


def _cache_exists(cache_dir):
    return (
        os.path.exists(f'{cache_dir}/splits.pkl') and
        os.path.exists(f'{cache_dir}/tokenizer')
    )


# ── Fonction principale ────────────────────────────────────

def load_data(n_train=100, n_val=20, n_test=20, seed=42,
              verbose=False, cache_dir=CACHE_DIR, force_reload=False):
    """
    Charge, sous-échantillonne et tokenise le dataset Emotion.
    Section 2.2 du PDF : sous-échantillonnage équilibré pour CPU.
    ✅ Cache automatique : si déjà chargé, retourne instantanément.

    Paramètres
    ----------
    n_train      : exemples par classe pour le train      (100 → 600 total)
    n_val        : exemples par classe pour la validation ( 20 → 120 total)
    n_test       : exemples par classe pour le test       ( 20 → 120 total)
    seed         : graine pour la reproductibilité
    force_reload : forcer le retéléchargement même si cache existe

    Retourne
    --------
    train_tokenized, val_tokenized, test_tokenized, tokenizer, label_names
    """

    def log(msg):
        if verbose:
            print(msg)

    # ── Vérifier le cache ──────────────────────────────────
    if _cache_exists(cache_dir) and not force_reload:
        return _load_cache(cache_dir)

    if force_reload:
        print("🔄 Rechargement forcé (force_reload=True)")

    # ── 1. Chargement ──────────────────────────────────────
    log("📥 Chargement du dataset Emotion Detection...")
    dataset     = load_dataset("dair-ai/emotion")
    label_names = dataset['train'].features['label'].names
    log(f"   Classes : {label_names}")

    # ── 2. Sous-échantillonnage équilibré (section 2.2) ────
    def create_subset(split, n_per_class):
        np.random.seed(seed)
        data    = dataset[split]
        indices = []
        for label_id in range(N_CLASSES):
            class_indices = [i for i, ex in enumerate(data) if ex['label'] == label_id]
            chosen        = np.random.choice(class_indices, size=n_per_class, replace=False)
            indices.extend(chosen.tolist())
        np.random.shuffle(indices)
        return data.select(indices)

    log("⚙️  Création des sous-ensembles équilibrés...")
    train_raw = create_subset('train',      n_train)
    val_raw   = create_subset('validation', n_val)
    test_raw  = create_subset('test',       n_test)

    log(f"   ✅ Train      : {len(train_raw)} exemples ({n_train}/classe)")
    log(f"   ✅ Validation : {len(val_raw)} exemples ({n_val}/classe)")
    log(f"   ✅ Test       : {len(test_raw)} exemples ({n_test}/classe)")

    # ── 3. Tokenisation ────────────────────────────────────
    log("📥 Chargement du tokenizer DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )

    log("⚙️  Tokenisation...")
    cols = ['input_ids', 'attention_mask', 'label']

    train_tokenized = train_raw.map(tokenize_function, batched=True)
    val_tokenized   = val_raw.map(tokenize_function,   batched=True)
    test_tokenized  = test_raw.map(tokenize_function,  batched=True)

    train_tokenized.set_format(type='torch', columns=cols)
    val_tokenized.set_format(  type='torch', columns=cols)
    test_tokenized.set_format( type='torch', columns=cols)

    log("✅ Données prêtes !\n")

    # ── 4. Sauvegarder le cache ────────────────────────────
    _save_cache(train_tokenized, val_tokenized, test_tokenized,
                tokenizer, label_names, cache_dir)

    return train_tokenized, val_tokenized, test_tokenized, tokenizer, label_names


# ── Fonction notebook ──────────────────────────────────────

def load_data_as_dataframe(n_train=100, n_val=20, n_test=20, seed=42,
                           cache_dir=CACHE_DIR, force_reload=False):
    """
    Charge les données et retourne un DataFrame lisible pour le notebook.
    ✅ Cache automatique — chargement instantané après la 1ère fois.

    Retourne
    --------
    train_data, val_data, test_data, tokenizer, label_names, train_df
    """

    train_data, val_data, test_data, tokenizer, label_names = load_data(
        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,
        verbose=False, cache_dir=cache_dir, force_reload=force_reload
    )

    # Construire le DataFrame pour l'affichage
    train_df = pd.DataFrame({
        'texte_original': [
            tokenizer.decode(train_data[i]['input_ids'], skip_special_tokens=True)
            for i in range(len(train_data))
        ],
        'label': [train_data[i]['label'].item() for i in range(len(train_data))],
    })
    train_df['emotion'] = train_df['label'].map(dict(enumerate(label_names)))

    print(f"\n📊 Dataset chargé : {len(train_data)} train | {len(val_data)} val | {len(test_data)} test")
    print(f"   Classes : {label_names}")

    return train_data, val_data, test_data, tokenizer, label_names, train_df
