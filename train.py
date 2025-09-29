
#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure TF uses CPU only

import argparse
import json
import hashlib
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

# CPU threading and XLA
num_threads = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)
try:
    tf.config.optimizer.set_jit(True)  # XLA
    jit_enabled = True
except Exception:
    jit_enabled = False

print(f"TensorFlow CPU mode. Threads: {num_threads}, XLA enabled: {jit_enabled}")

# -------------------------
# Utilities
# -------------------------
def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def convert_datasets_to_txt(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    converted = []
    for fname in os.listdir(data_dir):
        full = os.path.join(data_dir, fname)
        name, ext = os.path.splitext(fname.lower())
        out = os.path.join(data_dir, f"{name}.txt")
        if ext == ".txt" or os.path.exists(out):
            continue
        lines = []
        try:
            if ext in [".csv", ".tsv"]:
                sep = "," if ext == ".csv" else "\t"
                df = pd.read_csv(full, sep=sep, encoding="utf-8", header=None, on_bad_lines="skip")
                lines = df.iloc[:, -1].dropna().astype(str).tolist()
            elif ext == ".json":
                with open(full, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                for key in ("utterances","lines","text"):
                                    if key in item:
                                        if isinstance(item[key], list):
                                            lines.extend([str(x) for x in item[key]])
                                        else:
                                            lines.append(str(item[key]))
                    elif isinstance(data, dict):
                        for key in ("utterances","lines","text"):
                            if key in data:
                                if isinstance(data[key], list):
                                    lines.extend([str(x) for x in data[key]])
                                else:
                                    lines.append(str(data[key]))
            if lines:
                with open(out, "w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line.strip() + "\n")
                converted.append(out)
                print(f"Converted {fname} -> {out}")
        except Exception as e:
            print(f"Failed to convert {fname}: {e}")
    return converted

def load_corpus(data_dir="./data"):
    texts = []
    file_hashes = {}
    for fname in sorted(glob(os.path.join(data_dir, "*.txt"))):
        with open(fname, "r", encoding="utf-8") as f:
            texts.append(f.read())
        file_hashes[fname] = md5sum(fname)
    return " ".join(texts), file_hashes

def build_vocab(text, min_freq=1, max_vocab=50000):
    tokens = text.strip().split()
    freq = Counter(tokens)
    vocab = [tok for tok, c in freq.items() if c >= min_freq]
    vocab = sorted(vocab, key=lambda t: -freq[t])[:max_vocab]
    vocab = ["<PAD>", "<UNK>"] + vocab
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = {i: s for s, i in stoi.items()}
    return vocab, stoi, itos

def encode_to_ids(text, stoi):
    return np.array([stoi.get(tok, stoi["<UNK>"]) for tok in text.strip().split()], dtype=np.int32)

# -------------------------
# Model builder and helpers
# -------------------------
def build_model(vocab_size, emb_dim=128, context_size=2):
    inputs = tf.keras.Input(shape=(context_size*2,), dtype=tf.int32, name="context")
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, name="embed")(inputs)
    x = tf.keras.layers.Reshape((context_size*2*emb_dim,))(x)
    x = tf.keras.layers.Dense(units=emb_dim*2, activation="tanh", name="hid")(x)
    logits = tf.keras.layers.Dense(units=vocab_size, activation=None, name="logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

def rebuild_and_load_weights(old_vocab_size, new_vocab_size, emb_dim, context_size, weights_npz):
    """
    Build new model with new_vocab_size and transfer weights from the saved npz dict where possible.
    weights_npz: dict-like from np.load(..., allow_pickle=True)
    """
    # Build old model to know layer sizes (if needed)
    old_model = build_model(old_vocab_size, emb_dim=emb_dim, context_size=context_size)
    # instantiate variables
    dummy = np.zeros((1, context_size*2), dtype=np.int32)
    old_model(dummy, training=False)

    new_model = build_model(new_vocab_size, emb_dim=emb_dim, context_size=context_size)
    new_model(dummy, training=False)

    # Transfer embedding weights if present
    # weight keys are saved as var.name in our saving code (e.g. 'embed/embeddings:0', 'logits/kernel:0', 'logits/bias:0')
    saved = weights_npz
    try:
        old_emb = saved.get("embed/embeddings:0")
    except Exception:
        old_emb = None
    if old_emb is not None:
        new_emb = new_model.get_layer("embed").get_weights()[0]
        n_copy = min(old_emb.shape[0], new_emb.shape[0])
        new_emb[:n_copy] = old_emb[:n_copy]
        new_model.get_layer("embed").set_weights([new_emb])

    # Transfer logits kernel/bias if present
    try:
        old_kernel = saved.get("logits/kernel:0")
        old_bias = saved.get("logits/bias:0")
    except Exception:
        old_kernel = None
        old_bias = None

    try:
        logits_layer = new_model.get_layer("logits")
        new_kernel, new_bias = logits_layer.get_weights()
        if old_kernel is not None:
            hk_old, ov = old_kernel.shape
            hk_new, nv = new_kernel.shape
            hk_copy = min(hk_old, hk_new)
            nv_copy = min(ov, nv)
            new_kernel[:hk_copy, :nv_copy] = old_kernel[:hk_copy, :nv_copy]
        if old_bias is not None:
            nb_copy = min(old_bias.shape[0], new_bias.shape[0])
            new_bias[:nb_copy] = old_bias[:nb_copy]
        logits_layer.set_weights([new_kernel, new_bias])
    except Exception:
        pass

    return new_model

# -------------------------
# Dataset builder
# -------------------------
def make_dataset(token_ids, context_size=2, batch_size=2048, shuffle=True, cache=True):
    cs = context_size
    n = len(token_ids)
    contexts = []
    targets = []
    for i in range(cs, n-cs):
        left = token_ids[i-cs:i]
        right = token_ids[i+1:i+1+cs]
        context = np.concatenate([left, right])
        target = token_ids[i]
        contexts.append(context)
        targets.append(target)
    contexts = np.array(contexts, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((contexts, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(200000, len(contexts)))
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Save / load helpers
# -------------------------
def save_checkpoint(model, meta, checkpoint_prefix):
    # save weights as npz mapping var.name -> numpy array
    weights = {}
    for var in model.trainable_variables:
        weights[var.name] = var.numpy()
    np.savez(checkpoint_prefix + ".weights.npz", **weights)
    with open(checkpoint_prefix + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[Checkpoint] saved to {checkpoint_prefix}.*")

def load_weights_npz(path):
    return dict(np.load(path, allow_pickle=True))

# -------------------------
# Main flow
# -------------------------
def main(args):
    data_dir = args.data_dir
    ck_prefix = args.checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    context_size = args.context_size
    lr = args.lr
    max_vocab = args.max_vocab

    os.makedirs(data_dir, exist_ok=True)

    # convert any csv/json -> txt
    convert_datasets_to_txt(data_dir)

    text, file_hashes = load_corpus(data_dir)
    if not text.strip():
        print("No text data found in", data_dir)
        return

    meta_path = ck_prefix + ".meta.json"
    weights_path = ck_prefix + ".weights.npz"

    if os.path.exists(meta_path) and os.path.exists(weights_path):
        print("Found existing checkpoint. Loading metadata...")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        stoi = meta.get("stoi", {})
        itos = {int(k): v for k, v in meta.get("itos", {}).items()}
        old_vocab_size = len(stoi)
        print(f"Old vocab size: {old_vocab_size}")

        # detect new words
        tokens = text.strip().split()
        token_set = set(tokens)
        new_words = [w for w in token_set if w not in stoi]
        if new_words:
            print(f"Found {len(new_words)} new tokens; expanding vocab...")
            # append new words to stoi/itos
            for w in new_words:
                idx = len(stoi)
                stoi[w] = idx
                itos[idx] = w
        new_vocab_size = len(stoi)

        # load weight npz
        saved = load_weights_npz(weights_path)

        # rebuild model with new vocab and copy weights where possible
        model = rebuild_and_load_weights(old_vocab_size, new_vocab_size, emb_dim, context_size, saved)
        print("Model rebuilt and weights transferred (where possible).")
    else:
        print("No checkpoint found. Building new vocab and model from data...")
        vocab, stoi, itos = build_vocab(text, max_vocab=max_vocab)
        new_vocab_size = len(vocab)
        model = build_model(new_vocab_size, emb_dim=emb_dim, context_size=context_size)
        meta = {"stoi": stoi, "itos": itos, "file_hashes": {}}

    # prepare dataset
    token_ids = encode_to_ids(text, stoi)
    ds = make_dataset(token_ids, context_size=context_size, batch_size=batch_size, shuffle=True, cache=True)

    # compile model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["sparse_categorical_accuracy"])

    # callback to save at epoch end and update meta
    class CheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            meta_local = {"stoi": stoi, "itos": itos, "file_hashes": file_hashes}
            save_checkpoint(model, meta_local, ck_prefix)
            print(f"Epoch {epoch+1} finished. loss={logs.get('loss'):.4f} acc={logs.get('sparse_categorical_accuracy'):.4f}")

    steps_per_epoch = None
    # optionally compute steps_per_epoch to avoid infinite dataset if repeated; use dataset size // batch_size
    # compute dataset size quickly
    dataset_size = max(1, (len(token_ids) - 2*context_size))
    steps_per_epoch = max(1, dataset_size // batch_size)

    print(f"Training: epochs={epochs}, batch_size={batch_size}, steps_per_epoch={steps_per_epoch}")

    model.fit(ds, epochs=epochs, callbacks=[CheckpointCallback()], steps_per_epoch=steps_per_epoch)

    # final save
    meta_out = {"stoi": stoi, "itos": itos, "file_hashes": file_hashes}
    save_checkpoint(model, meta_out, ck_prefix)
    print("Training complete. Saved checkpoint and metadata.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="toy_tf_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_vocab", type=int, default=50000)
    args = parser.parse_args()
    main(args)
