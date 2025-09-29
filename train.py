
#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force TensorFlow to use CPU only
import argparse
import json
import hashlib
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

# -------------------------
# Utilities
# -------------------------
def md5sum(filename):
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def convert_datasets_to_txt(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    converted_files = []
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        name, ext = os.path.splitext(fname.lower())
        out_file = os.path.join(data_dir, f"{name}.txt")
        if ext == ".txt" or os.path.exists(out_file):
            continue
        lines = []
        try:
            if ext in [".csv", ".tsv"]:
                sep = "," if ext == ".csv" else "\t"
                df = pd.read_csv(full_path, sep=sep, encoding="utf-8", header=None, on_bad_lines="skip")
                # use last column as text candidate
                lines = df.iloc[:, -1].dropna().astype(str).tolist()
            elif ext == ".json":
                with open(full_path, "r", encoding="utf-8") as f:
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
                with open(out_file, "w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line.strip() + "\n")
                converted_files.append(out_file)
                print(f"Converted {fname} -> {out_file}")
        except Exception as e:
            print(f"Failed to convert {fname}: {e}")
    return converted_files

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
# Model helpers
# -------------------------
def build_model(vocab_size, emb_dim=64, context_size=2):
    """
    Simple CBOW-like model: embed context tokens, flatten, linear to vocab logits.
    """
    inputs = tf.keras.Input(shape=(context_size*2,), dtype=tf.int32, name="context")
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, name="embed")(inputs)
    x = tf.keras.layers.Reshape((context_size*2*emb_dim,))(x)
    x = tf.keras.layers.Dense(units=emb_dim*2, activation="tanh")(x)
    logits = tf.keras.layers.Dense(units=vocab_size, activation=None, name="logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

def rebuild_and_load_weights(old_model, new_vocab_size, checkpoint_weights):
    """
    Rebuild a model with new_vocab_size and copy weights from old model where possible.
    checkpoint_weights is a dict mapping layer name -> numpy array weights
    """
    new_model = build_model(new_vocab_size,
                            emb_dim=old_model.get_layer("embed").output_shape[-1],
                            context_size=(old_model.input_shape[1] // 2))
    # Build model by calling with dummy input
    dummy = np.zeros((1, new_model.input_shape[1]), dtype=np.int32)
    new_model(dummy, training=False)

    # Copy embedding weights
    old_emb_w = checkpoint_weights.get("embed/embeddings:0")
    if old_emb_w is not None:
        # old_emb_w shape (old_vocab, emb_dim)
        emb_layer = new_model.get_layer("embed")
        new_emb_w = emb_layer.get_weights()[0]
        n_copy = min(old_emb_w.shape[0], new_emb_w.shape[0])
        new_emb_w[:n_copy] = old_emb_w[:n_copy]
        emb_layer.set_weights([new_emb_w])

    # Copy logits Dense weights (kernel and bias)
    old_logits_kernel = checkpoint_weights.get("logits/kernel:0")
    old_logits_bias = checkpoint_weights.get("logits/bias:0")
    try:
        logits_layer = new_model.get_layer("logits")
        new_kernel, new_bias = logits_layer.get_weights()
        if old_logits_kernel is not None:
            # old shape (hidden, old_vocab)
            hk_old, ov = old_logits_kernel.shape
            hk_new, nv = new_kernel.shape
            # copy matching rows/cols
            hk_copy = min(hk_old, hk_new)
            nv_copy = min(ov, nv)
            new_kernel[:hk_copy, :nv_copy] = old_logits_kernel[:hk_copy, :nv_copy]
        if old_logits_bias is not None:
            nb_copy = min(old_logits_bias.shape[0], new_bias.shape[0])
            new_bias[:nb_copy] = old_logits_bias[:nb_copy]
        logits_layer.set_weights([new_kernel, new_bias])
    except Exception:
        pass

    return new_model

# -------------------------
# Dataset builder
# -------------------------
def make_dataset(token_ids, context_size=2, batch_size=512, shuffle=True):
    """
    token_ids: 1D numpy array of token ids
    Produces (context, target) pairs where context_size is number of tokens on each side.
    """
    cs = context_size
    n = len(token_ids)
    contexts = []
    targets = []
    for i in range(cs, n - cs):
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
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Main flow
# -------------------------
def main(args):
    data_dir = args.data_dir
    checkpoint_prefix = args.checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    context_size = args.context_size
    lr = args.lr

    os.makedirs(data_dir, exist_ok=True)

    # Step 1: convert datasets to txt
    convert_datasets_to_txt(data_dir)

    # Step 2: load corpus
    text, file_hashes = load_corpus(data_dir)
    if not text.strip():
        print("No text data found in", data_dir)
        return

    # Step 3: load existing checkpoint metadata if present
    meta_path = checkpoint_prefix + ".meta.json"
    weights_path = checkpoint_prefix + ".weights.npz"
    stoi = None
    itos = None
    model = None

    if os.path.exists(meta_path) and os.path.exists(weights_path):
        print("Loading existing checkpoint metadata...")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        stoi = meta["stoi"]
        itos = {int(k): v for k, v in meta["itos"].items()}
        old_vocab_size = len(stoi)

        # build token ids for new corpus
        tokens = text.strip().split()
        token_set = set(tokens)
        new_words = [w for w in token_set if w not in stoi]
        if new_words:
            print(f"Found {len(new_words)} new words — expanding vocab...")
            # extend stoi/itos
            for w in new_words:
                idx = len(stoi)
                stoi[w] = idx
                itos[idx] = w
            new_vocab_size = len(stoi)
        else:
            new_vocab_size = old_vocab_size

        # load saved weights (npz)
        npz = np.load(weights_path, allow_pickle=True)
        saved_weights = dict(npz)

        # build model with new vocab size and copy weights
        # For safe building, construct old model to inspect layer sizes
        old_model = build_model(old_vocab_size, emb_dim=emb_dim, context_size=context_size)
        # create old model variables by calling once
        dummy = np.zeros((1, context_size*2), dtype=np.int32)
        old_model(dummy, training=False)
        model = rebuild_and_load_weights(old_model, new_vocab_size, saved_weights)
    else:
        # Create new vocab + model
        print("Creating new model and vocab...")
        vocab, stoi, itos = build_vocab(text, max_vocab=args.max_vocab)
        new_vocab_size = len(vocab)
        model = build_model(new_vocab_size, emb_dim=emb_dim, context_size=context_size)

    # Prepare data
    token_ids = encode_to_ids(text, stoi)
    ds = make_dataset(token_ids, context_size=context_size, batch_size=batch_size, shuffle=True)

    # Compile model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["sparse_categorical_accuracy"])

    # Callback to save weights each epoch
    class SaveWeightsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # save weights to npz
            weights = {}
            for var in model.trainable_variables:
                weights[var.name] = var.numpy()
            np.savez(checkpoint_prefix + ".weights.npz", **weights)
            # save meta
            meta = {"stoi": stoi, "itos": itos, "file_hashes": file_hashes}
            with open(checkpoint_prefix + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
            print(f"[Checkpoint] saved weights + meta (epoch {epoch+1})")

    # Fit
    model.fit(ds, epochs=epochs, callbacks=[SaveWeightsCallback()])

    # Final save
    weights = {}
    for var in model.trainable_variables:
        weights[var.name] = var.numpy()
    np.savez(checkpoint_prefix + ".weights.npz", **weights)
    meta = {"stoi": stoi, "itos": itos, "file_hashes": file_hashes}
    with open(checkpoint_prefix + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Training complete. Model + meta saved to:", checkpoint_prefix + ".*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="toy_tf_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_vocab", type=int, default=50000)
    args = parser.parse_args()
    main(args)
