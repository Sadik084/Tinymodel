#!/usr/bin/env python3
# train_fast.py
"""
Faster TinyLM training using:
 - float32 for weights/activations
 - BLAS thread control via env vars
 - Numba JIT for heavy forward/backward & embedding scatter-add
"""

import os
# Set these BEFORE importing numpy so BLAS/MKL picks them up
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 1)
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 1)

import json
import hashlib
from glob import glob
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd

# Try to import numba; if unavailable, we proceed but slower
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------
def md5sum(filename):
    with open(filename, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def convert_datasets_to_txt(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    converted_files = []

    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        name, ext = os.path.splitext(fname.lower())
        out_file = os.path.join(data_dir, f"{name}.txt")

        if os.path.exists(out_file) or ext == ".txt":
            continue

        lines = []
        try:
            if ext in [".csv", ".tsv"]:
                sep = "," if ext == ".csv" else "\t"
                df = pd.read_csv(full_path, sep=sep, encoding="utf-8", header=None, on_bad_lines='skip')
                lines = df.iloc[:, -1].dropna().astype(str).tolist()

            elif ext == ".json":
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                for key in ["utterances", "lines", "text"]:
                                    if key in item:
                                        if isinstance(item[key], list):
                                            lines.extend([str(x) for x in item[key]])
                                        else:
                                            lines.append(str(item[key]))
                    elif isinstance(data, dict):
                        for key in ["utterances", "lines", "text"]:
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
                print(f"Converted {fname} → {out_file}")
        except Exception as e:
            print(f"Failed to convert {fname}: {e}")

    return converted_files

def load_corpus(data_dir):
    texts = []
    file_hashes = {}
    for fname in glob(os.path.join(data_dir, "*.txt")):
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

def encode(text, stoi):
    return np.array([stoi.get(tok, stoi["<UNK>"]) for tok in text.strip().split()], dtype=np.int64)

# -------------------------
# Model (lightweight)
# -------------------------
class TinyLM:
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=256, context_size=4, dtype=np.float32):
        self.dtype = dtype
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_size = int(context_size)

        # Initialize in float32
        self.W_emb = (np.random.randn(self.vocab_size, self.emb_dim).astype(self.dtype)) * 0.01
        self.W1 = (np.random.randn(self.context_size * self.emb_dim, self.hidden_dim).astype(self.dtype)) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim), dtype=self.dtype)
        self.W2 = (np.random.randn(self.hidden_dim, self.vocab_size).astype(self.dtype)) * 0.01
        self.b2 = np.zeros((1, self.vocab_size), dtype=self.dtype)

    def expand_vocab(self, new_words):
        add = len(new_words)
        if add == 0:
            return
        new_W_emb = (np.random.randn(add, self.emb_dim).astype(self.dtype)) * 0.01
        self.W_emb = np.vstack([self.W_emb, new_W_emb])

        new_W2 = (np.random.randn(self.hidden_dim, add).astype(self.dtype)) * 0.01
        self.W2 = np.hstack([self.W2, new_W2])

        new_b2 = np.zeros((1, add), dtype=self.dtype)
        self.b2 = np.hstack([self.b2, new_b2])

        self.vocab_size += add
        print(f"Expanded model vocabulary by {add} words → New vocab size: {self.vocab_size}")

# -------------------------
# Numba-accelerated batch gradients
# -------------------------
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def batch_grads_numba(W_emb, W1, b1, W2, b2, Xb, Yb, context_size, emb_dim, hidden_dim):
        """
        Compute logits, loss and gradients for a single batch.
        Returns:
         loss (float), dW_emb, dW1, db1, dW2, db2
        """
        N = Xb.shape[0]
        vocab_size = W_emb.shape[0]

        # Embedding lookup -> (N, context_size*emb_dim)
        emb_mat = np.empty((N, context_size * emb_dim), dtype=W_emb.dtype)
        for i in prange(N):
            row = emb_mat[i]
            pos = 0
            for j in range(context_size):
                idx = Xb[i, j]
                for k in range(emb_dim):
                    row[pos + k] = W_emb[idx, k]
                pos += emb_dim

        # Hidden activations
        h = np.empty((N, hidden_dim), dtype=W1.dtype)
        for i in prange(N):
            # h = tanh(emb @ W1 + b1)
            for j in range(hidden_dim):
                s = b1[0, j]
                for k in range(context_size * emb_dim):
                    s += emb_mat[i, k] * W1[k, j]
                # tanh
                h[i, j] = (np.tanh(s))

        # logits = h @ W2 + b2
        logits = np.empty((N, vocab_size), dtype=W2.dtype)
        for i in prange(N):
            for j in range(vocab_size):
                s = b2[0, j]
                for k in range(hidden_dim):
                    s += h[i, k] * W2[k, j]
                logits[i, j] = s

        # softmax -> probs
        probs = np.empty_like(logits)
        loss = 0.0
        for i in prange(N):
            # numeric stabilization
            m = logits[i, 0]
            for j in range(1, vocab_size):
                if logits[i, j] > m:
                    m = logits[i, j]

            denom = 0.0
            for j in range(vocab_size):
                denom += np.exp(logits[i, j] - m)

            for j in range(vocab_size):
                probs[i, j] = np.exp(logits[i, j] - m) / denom

            y = Yb[i]
            # accumulate negative log likelihood
            loss -= np.log(probs[i, y] + 1e-12)

        loss = loss / N

        # dlogits
        dlogits = np.empty_like(probs)
        for i in prange(N):
            for j in range(vocab_size):
                dlogits[i, j] = probs[i, j]
            dlogits[i, Yb[i]] -= 1.0
        for i in prange(N):
            for j in range(vocab_size):
                dlogits[i, j] /= N

        # dW2 = h.T @ dlogits
        dW2 = np.zeros_like(W2)
        for i in prange(hidden_dim):
            for j in range(vocab_size):
                s = 0.0
                for n in range(N):
                    s += h[n, i] * dlogits[n, j]
                dW2[i, j] = s

        # db2
        db2 = np.zeros((1, vocab_size), dtype=W2.dtype)
        for j in prange(vocab_size):
            s = 0.0
            for n in range(N):
                s += dlogits[n, j]
            db2[0, j] = s

        # dh = dlogits @ W2.T
        dh = np.empty((N, hidden_dim), dtype=W2.dtype)
        for n in prange(N):
            for i in range(hidden_dim):
                s = 0.0
                for j in range(vocab_size):
                    s += dlogits[n, j] * W2[i, j]
                dh[n, i] = s

        # dh_raw = dh * (1 - h^2)
        dh_raw = np.empty_like(dh)
        for n in prange(N):
            for i in range(hidden_dim):
                dh_raw[n, i] = dh[n, i] * (1.0 - h[n, i] * h[n, i])

        # dW1 = emb_mat.T @ dh_raw  -> shape (context*emb_dim, hidden_dim)
        dW1 = np.zeros_like(W1)
        for i in prange(context_size * emb_dim):
            for j in range(hidden_dim):
                s = 0.0
                for n in range(N):
                    s += emb_mat[n, i] * dh_raw[n, j]
                dW1[i, j] = s

        # db1
        db1 = np.zeros((1, hidden_dim), dtype=W1.dtype)
        for j in prange(hidden_dim):
            s = 0.0
            for n in range(N):
                s += dh_raw[n, j]
            db1[0, j] = s

        # demb = dh_raw @ W1.T  -> (N, context*emb_dim)
        demb_flat = np.empty((N, context_size * emb_dim), dtype=W1.dtype)
        for n in prange(N):
            for i in range(context_size * emb_dim):
                s = 0.0
                for j in range(hidden_dim):
                    s += dh_raw[n, j] * W1[i, j]
                demb_flat[n, i] = s

        # reshape demb and scatter-add into dW_emb
        dW_emb = np.zeros_like(W_emb)
        for n in prange(N):
            for c in range(context_size):
                idx = Xb[n, c]
                base = c * emb_dim
                for k in range(emb_dim):
                    dW_emb[idx, k] += demb_flat[n, base + k]

        return loss, dW_emb, dW1, db1, dW2, db2

# -------------------------
# NumPy fallback (no numba)
# -------------------------
def batch_grads_numpy(W_emb, W1, b1, W2, b2, Xb, Yb, context_size, emb_dim, hidden_dim):
    # Numpy version: same math but using numpy ops (works without numba)
    N = Xb.shape[0]
    # Embedding lookup
    emb = W_emb[Xb].reshape(N, context_size * emb_dim).astype(W_emb.dtype)  # (N, context*emb)
    h = np.tanh(emb.dot(W1) + b1)  # (N, hidden)
    logits = h.dot(W2) + b2  # (N, vocab)
    # softmax
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits_stable)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(N), Yb] + 1e-12).mean()
    # dlogits
    dlogits = probs.copy()
    dlogits[np.arange(N), Yb] -= 1.0
    dlogits /= N
    dW2 = h.T.dot(dlogits)
    db2 = dlogits.sum(axis=0, keepdims=True)
    dh = dlogits.dot(W2.T)
    dh_raw = dh * (1 - h * h)
    dW1 = emb.T.dot(dh_raw)
    db1 = dh_raw.sum(axis=0, keepdims=True)
    demb = dh_raw.dot(W1.T).reshape(N, context_size, emb_dim)
    dW_emb = np.zeros_like(W_emb)
    for i in range(context_size):
        np.add.at(dW_emb, Xb[:, i], demb[:, i])
    return loss, dW_emb, dW1, db1, dW2, db2

# -------------------------
# Batching
# -------------------------
def make_batches(data, context_size, batch_size=512):
    X, Y = [], []
    for i in range(len(data) - context_size):
        X.append(data[i:i+context_size])
        Y.append(data[i+context_size])
    X = np.array(X, dtype=np.int64)
    Y = np.array(Y, dtype=np.int64)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = idx[i:i+batch_size]
        yield X[j], Y[j]

# -------------------------
# Training
# -------------------------
def train(model, data, epochs=5, lr=0.005, batch_size=512, use_numba=NUMBA_AVAILABLE):
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        batches = list(make_batches(data, model.context_size, batch_size))
        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
        for Xb, Yb in pbar:
            # choose computation function
            if use_numba and NUMBA_AVAILABLE:
                loss, dW_emb, dW1, db1, dW2, db2 = batch_grads_numba(
                    model.W_emb, model.W1, model.b1, model.W2, model.b2,
                    Xb, Yb, model.context_size, model.emb_dim, model.hidden_dim
                )
            else:
                loss, dW_emb, dW1, db1, dW2, db2 = batch_grads_numpy(
                    model.W_emb, model.W1, model.b1, model.W2, model.b2,
                    Xb, Yb, model.context_size, model.emb_dim, model.hidden_dim
                )

            # Gradient descent updates (in-place)
            model.W_emb -= lr * dW_emb
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            model.W2 -= lr * dW2
            model.b2 -= lr * db2

            total_loss += float(loss)
            steps += 1
            pbar.set_postfix({"loss": f"{total_loss/steps:.4f}"})
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/steps:.6f}")

# -------------------------
# Main Auto-Update Training with Dynamic Expansion
# -------------------------
def main(data_dir="./data", checkpoint="toy_model_fast.npz", epochs=5, emb_dim=64, hidden_dim=256, context_size=4):
    os.makedirs(data_dir, exist_ok=True)

    # Convert CSV/JSON datasets to txt
    convert_datasets_to_txt(data_dir)

    # Load corpus
    text, file_hashes = load_corpus(data_dir)
    if not text.strip():
        print("No text data found in ./data folder.")
        return

    # Load or create model
    if os.path.exists(checkpoint):
        print("🔄 Loading existing model...")
        weights = np.load(checkpoint, allow_pickle=True)
        meta = json.loads(str(weights["meta"]))
        stoi = meta["stoi"]
        itos = {int(k): v for k, v in meta["itos"].items()}
        vocab_size = len(stoi)

        model = TinyLM(vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, context_size=context_size)
        model.W_emb = weights["W_emb"].astype(model.dtype)
        model.W1 = weights["W1"].astype(model.dtype)
        model.b1 = weights["b1"].astype(model.dtype)
        model.W2 = weights["W2"].astype(model.dtype)
        model.b2 = weights["b2"].astype(model.dtype)

        tokens = set(text.strip().split())
        new_words = [tok for tok in tokens if tok not in stoi]
        if new_words:
            model.expand_vocab(new_words)
            for i, tok in enumerate(new_words):
                idx = vocab_size + i
                stoi[tok] = idx
                itos[idx] = tok

        data = encode(text, stoi)
        train(model, data, epochs=epochs, lr=0.005, batch_size=512)

        meta["file_hashes"] = file_hashes
    else:
        print("✨ Creating new model...")
        vocab, stoi, itos = build_vocab(text, max_vocab=50000)
        model = TinyLM(len(vocab), emb_dim=emb_dim, hidden_dim=hidden_dim, context_size=context_size)
        data = encode(text, stoi)
        train(model, data, epochs=epochs, lr=0.005, batch_size=512)

        meta = {"stoi": stoi, "itos": itos, "file_hashes": file_hashes}

    # Save model + meta
    meta_str = json.dumps(meta)
    np.savez(checkpoint,
             W_emb=model.W_emb.astype(np.float32), W1=model.W1.astype(np.float32), b1=model.b1.astype(np.float32),
             W2=model.W2.astype(np.float32), b2=model.b2.astype(np.float32),
             meta=np.array(meta_str))
    print("✅ Model saved:", checkpoint)

if __name__ == "__main__":
    print("NUMBA_AVAILABLE =", NUMBA_AVAILABLE)
    main(epochs=5, emb_dim=64, hidden_dim=256, context_size=4)
