import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────
batch_size    = 16      # sequences per batch
block_size    = 64      # context length: model sees 64 characters at a time
max_iters     = 3000    # total training steps
eval_every    = 300     # print loss at 0, 300, 600, ..., 3000
learning_rate = 1e-3
n_embd        = 64      # embedding dimension
n_head        = 4       # number of attention heads
n_layer       = 4       # number of transformer blocks
dropout       = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load text corpus ──────────────────────────────────────────────────────────
with open('wizard of oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ── Character-level vocabulary ────────────────────────────────────────────────
chars      = sorted(set(text))
vocab_size = len(chars)                        # 67 unique characters
stoi = {ch: i for i, ch in enumerate(chars)}  # char → index
itos = {i: ch for i, ch in enumerate(chars)}  # index → char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Train / val split ─────────────────────────────────────────────────────────
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── Batch sampler ─────────────────────────────────────────────────────────────
def get_batch(split):
    """Return a random (input, target) batch from train or val data."""
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i + block_size]         for i in ix])
    y  = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── Loss estimator ────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, eval_batches=50):
    """Average loss over multiple batches for a stable estimate."""
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ── Causal self-attention head ────────────────────────────────────────────────
class Head(nn.Module):
    """Single causal attention head: each token only attends to past tokens."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # causal mask: lower-triangular so position i cannot see position j > i
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # attention scores, scaled by sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v     # (B, T, head_size)

# ── Multi-head attention ──────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """Run n_head attention heads in parallel, then project back to n_embd."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

# ── Feed-forward block ────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    """Position-wise MLP applied after attention: expand → GELU → contract."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ── Transformer block ─────────────────────────────────────────────────────────
class Block(nn.Module):
    """One transformer block: layer-norm → attention → residual,
       then layer-norm → feed-forward → residual."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)  # self-attention
        self.ffwd = FeedForward(n_embd)                    # feed-forward
        self.ln1  = nn.LayerNorm(n_embd)                   # pre-norm before attention
        self.ln2  = nn.LayerNorm(n_embd)                   # pre-norm before FFN

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # attention with residual connection
        x = x + self.ffwd(self.ln2(x))  # FFN with residual connection
        return x

# ── SmallGPT model ────────────────────────────────────────────────────────────
class SmallGPT(nn.Module):
    """
    Small GPT: token embedding + positional embedding → n_layer Transformer
    blocks → LayerNorm → linear head projecting to vocab logits.
    """
    def __init__(self):
        super().__init__()
        # token embedding: maps each token id to a learned vector
        self.token_embedding  = nn.Embedding(vocab_size, n_embd)
        # position embedding: gives the model a sense of order within the context
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # stack of transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        # final layer norm before the output projection
        self.ln_f   = nn.LayerNorm(n_embd)
        # output head: project from n_embd to vocab logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                             # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, n_embd)
        x       = tok_emb + pos_emb                                    # (B, T, n_embd)
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)                                      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively sample max_new_tokens characters."""
        for _ in range(max_new_tokens):
            # crop context to block_size so position embedding doesn't overflow
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits     = logits[:, -1, :]              # last time-step logits
            probs      = F.softmax(logits, dim=-1)
            idx_next   = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat((idx, idx_next), dim=1)
        return idx

# ── Build model & optimizer ───────────────────────────────────────────────────
torch.manual_seed(1337)
model     = SmallGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_params = sum(p.numel() for p in model.parameters())

print(f"Training on {device}  |  vocab_size={vocab_size}  |  "
      f"params={total_params:,}  |  steps={max_iters}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, val_losses, loss_steps = [], [], []

for step in range(max_iters + 1):

    # record and print train / val loss at 0, 300, 600, ..., 3000
    if step % eval_every == 0:
        losses = estimate_loss(model)
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        loss_steps.append(step)
        print(f"step {step:>4d}  |  train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

    if step == max_iters:
        break

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ── Plot train / val loss ─────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(loss_steps, train_losses, label='Train loss')
plt.plot(loss_steps, val_losses,   label='Val loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('SmallGPT Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

# ── Generate sample text ──────────────────────────────────────────────────────
print("\n── Generated text ───────────────────────────────────────────────────────")
context   = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
