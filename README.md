# smallGPT

A character-level GPT built from scratch in PyTorch, trained on *The Wizard of Oz* text.

Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) and [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## Overview

This project walks through building a small GPT language model step by step, from raw text to a working Transformer that generates readable English prose.

The model is trained at the **character level** on *The Wizard of Oz* (~204k characters, vocab size 67) and learns to generate text that resembles the style of the book.

---

## Project Structure

```
smallGPT/
├── build_your_smallGPT.ipynb   # Step-by-step notebook (tokenization → attention → SmallGPT)
├── smallGPT.py                 # Clean standalone training script
├── wizard of oz.txt            # Training corpus
├── LICENSE
└── README.md
```

---

## What's Inside the Notebook

| Section | Description |
|---|---|
| Data loading | Read and explore the raw text |
| Tokenization | Character-level, word-level, and BPE (via HuggingFace & tiktoken) |
| Dataset & DataLoader | Sliding-window `(input, target)` pairs |
| Embeddings | Token and positional embeddings |
| Attention | `SelfAttention` → `CausalAttention` → `MultiHeadAttention` |
| Components | `LayerNorm`, `GELU`, `FeedForward` |
| Bigram baseline | Simplest possible language model |
| **SmallGPT** | Full Transformer: 4 layers, 4 heads, 64-dim embeddings |
| Loss curves | Matplotlib plot of train/val loss over 3000 steps |
| Text generation | Sample 500 characters from the trained model |

---

## Model Architecture

```
Input tokens
    │
    ▼
Token Embedding (67 → 64)  +  Positional Embedding (64 → 64)
    │
    ▼
Transformer Block × 4
  ├─ LayerNorm
  ├─ Causal Multi-Head Attention (4 heads, head_dim=16)
  ├─ Residual connection
  ├─ LayerNorm
  ├─ FeedForward (64 → 256 → 64, GELU)
  └─ Residual connection
    │
    ▼
LayerNorm → Linear (64 → 67) → Logits
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| `vocab_size` | 67 |
| `block_size` (context) | 64 |
| `n_embd` | 64 |
| `n_head` | 4 |
| `n_layer` | 4 |
| `dropout` | 0.1 |
| `batch_size` | 16 |
| `max_iters` | 3000 |
| `learning_rate` | 1e-3 |
| Total parameters | ~212k |

---

## Results

Training on a single GPU for 3000 steps:

```
step    0  |  train loss 4.3943  |  val loss 4.3969
step  300  |  train loss 2.2718  |  val loss 2.3042
step  600  |  train loss 2.1238  |  val loss 2.1646
step  900  |  train loss 1.8763  |  val loss 1.9353
step 1200  |  train loss 1.7530  |  val loss 1.8156
step 1500  |  train loss 1.6232  |  val loss 1.7224
step 1800  |  train loss 1.5779  |  val loss 1.6353
step 2100  |  train loss 1.4944  |  val loss 1.5989
step 2400  |  train loss 1.4614  |  val loss 1.5826
step 2700  |  train loss 1.4089  |  val loss 1.5335
step 3000  |  train loss 1.3772  |  val loss 1.5185
```

**Sample generated text:**

```
"Can my helf that Kas be to her sighte was, "Con just excroses on trock;
yefor over I would getterle was there Scrow, pids, pretent anconling and
dreecte and and I walit us tz a drown had where road mige send us moverce
harson to rected of therr would the Tin Woodman.
```

---

## Requirements

```
torch
tiktoken
tokenizers
matplotlib
```

Install with:

```bash
pip install torch tiktoken tokenizers matplotlib
```

---

## Usage

**Run the standalone training script:**

```bash
python smallGPT.py
```

This will train for 3000 steps, print loss every 300 steps, display a loss curve, and generate 500 characters of text.

**Or follow the notebook step by step:**

```bash
jupyter notebook build_your_smallGPT.ipynb
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The training text (*The Wizard of Oz* by L. Frank Baum, 1900) is in the public domain,
sourced from [Project Gutenberg](https://www.gutenberg.org/).
