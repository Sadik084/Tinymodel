import os
import json
import hashlib
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
from tqdm import tqdm

# ==========================================================
# Utilities
# ==========================================================
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

def build_vocab(text, min_freq=1, max_vocab=5000):
    tokens = text.strip().split()
    freq = Counter(tokens)
    vocab = [tok for tok, c in freq.items() if c >= min_freq]
    vocab = sorted(vocab, key=lambda t: -freq[t])[:max_vocab]
    vocab = ["<PAD>", "<UNK>"] + vocab
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = {i: s for s, i in stoi.items()}
    return vocab, stoi, itos

def encode(text, stoi):
    return [stoi.get(tok, stoi["<UNK>"]) for tok in text.strip().split()]

# ==========================================================
# Model
# ==========================================================
class TinyLM:
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64, context_size=2):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        self.W_emb = np.random.randn(vocab_size, emb_dim) * 0.01
        self.W1 = np.random.randn(context_size * emb_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b2 = np.zeros((1, vocab_size))

    def forward(self, X):
        emb = self.W_emb[X].reshape(X.shape[0], -1)
        h = np.tanh(emb @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits, h

    def expand_vocab(self, new_words):
        new_size = self.vocab_size + len(new_words)
        # Expand embedding
        new_W_emb = np.random.randn(len(new_words), self.emb_dim) * 0.01
        self.W_emb = np.vstack([self.W_emb, new_W_emb])
        # Expand output layer
        new_W2 = np.random.randn(self.hidden_dim, len(new_words)) * 0.01
        self.W2 = np.hstack([self.W2, new_W2])
        new_b2 = np.zeros((1, len(new_words)))
        self.b2 = np.hstack([self.b2, new_b2])
        self.vocab_size = new_size
        print(f"Expanded model vocabulary by {len(new_words)} words → New vocab size: {self.vocab_size}")

# ==========================================================
# Training utils
# ==========================================================
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

def cross_entropy(logits, y):
    probs = softmax(logits)
    N = y.shape[0]
    loss = -np.log(probs[np.arange(N), y] + 1e-9).mean()
    return loss, probs

def make_batches(data, context_size, batch_size=32):
    X, Y = [], []
    for i in range(len(data) - context_size):
        X.append(data[i:i+context_size])
        Y.append(data[i+context_size])
    X, Y = np.array(X), np.array(Y)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = idx[i:i+batch_size]
        yield X[j], Y[j]

def train(model, data, epochs=5, lr=0.01, batch_size=32):
    for epoch in range(epochs):
        total_loss, steps = 0, 0
        batches = list(make_batches(data, model.context_size, batch_size))
        for Xb, Yb in tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            logits, h = model.forward(Xb)
            loss, probs = cross_entropy(logits, Yb)

            N = Yb.shape[0]
            dlogits = probs
            dlogits[np.arange(N), Yb] -= 1
            dlogits /= N

            dW2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0, keepdims=True)

            dh = dlogits @ model.W2.T
            dh_raw = dh * (1 - h**2)

            dW1 = (model.W_emb[Xb].reshape(N, -1)).T @ dh_raw
            db1 = dh_raw.sum(axis=0, keepdims=True)

            demb = dh_raw @ model.W1.T
            demb = demb.reshape(N, model.context_size, model.emb_dim)

            dW_emb = np.zeros_like(model.W_emb)
            for i in range(model.context_size):
                np.add.at(dW_emb, Xb[:, i], demb[:, i])

            for param, grad in zip(
                [model.W_emb, model.W1, model.b1, model.W2, model.b2],
                [dW_emb, dW1, db1, dW2, db2],
            ):
                param -= lr * grad

            total_loss += loss
            steps += 1
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/steps:.4f}")

# ==========================================================
# Main Auto-Update Training with Dynamic Expansion
# ==========================================================
def main(data_dir="./data", checkpoint="toy_model.npz", epochs=10):
    os.makedirs(data_dir, exist_ok=True)

    # 1️⃣ Convert CSV/JSON datasets to txt
    convert_datasets_to_txt(data_dir)

    # 2️⃣ Load corpus
    text, file_hashes = load_corpus(data_dir)
    if not text.strip():
        print("No text data found in ./data folder.")
        return

    # 3️⃣ Load existing model or create new
    if os.path.exists(checkpoint):
        print("🔄 Updating existing model...")
        weights = np.load(checkpoint, allow_pickle=True)
        meta = json.loads(str(weights["meta"]))
        stoi, itos = meta["stoi"], {int(k): v for k, v in meta["itos"].items()}
        vocab_size = len(stoi)

        model = TinyLM(vocab_size)
        model.W_emb = weights["W_emb"]
        model.W1 = weights["W1"]
        model.b1 = weights["b1"]
        model.W2 = weights["W2"]
        model.b2 = weights["b2"]

        # Check for new words in data
        tokens = set(text.strip().split())
        new_words = [tok for tok in tokens if tok not in stoi]
        if new_words:
            model.expand_vocab(new_words)
            for i, tok in enumerate(new_words):
                idx = vocab_size + i
                stoi[tok] = idx
                itos[idx] = tok

        data = encode(text, stoi)
        train(model, data, epochs=epochs, lr=0.01, batch_size=64)

        meta["file_hashes"] = file_hashes
    else:
        print("✨ No model found, creating new...")
        vocab, stoi, itos = build_vocab(text)
        model = TinyLM(len(vocab))
        data = encode(text, stoi)
        train(model, data, epochs=epochs, lr=0.01, batch_size=64)

        meta = {"stoi": stoi, "itos": itos, "file_hashes": file_hashes}

    # 4️⃣ Save model + metadata
    meta_str = json.dumps(meta)
    np.savez(checkpoint,
             W_emb=model.W_emb, W1=model.W1, b1=model.b1,
             W2=model.W2, b2=model.b2,
             meta=np.array(meta_str))
    print("✅ Model saved:", checkpoint)

if __name__ == "__main__":
    main()
