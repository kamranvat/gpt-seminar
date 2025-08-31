import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import time

# from save_utils import save_checkpoint
from GPT_encode import GPTEncoder

# from GPT_generate import VOCAB_PATH

# ----------------------------
# Set K, encode, export .bin/.txt
# ----------------------------
K = 5
# recreate gpt_bin files if not exist (for convenience, TODO remove?)
if __name__ == "__main__":
    encoder = GPTEncoder(k=K)
    encoder.export(data_dir="gpt_bin")

# ----------------------------
# Paths / data format / saving
# ----------------------------
TRAIN_IDS_TXT = Path("gpt_bin/train.txt")  # whitespace-delimited integers
VAL_IDS_TXT = Path("gpt_bin/val.txt")  # whitespace-delimited integers
VOCAB_TOKENS_TXT = Path(
    f"data/vocab_nNone_k{K}.txt"
)  # JSON list of strings; set to None to disable decoding
SAVE_INTERVAL = 4000

# ----------------------------
# Train from checkpoint (optional)
# ----------------------------
# WARNING if there is no checkpoint, do not run the next lines
# (might crash GPU memory)
# NOTE remember to uncomment checkpoint loading in the train loop, too
# if you want to resume training from a checkpoint, specify the path here
# (else, comment out the next three lines)
# CHECKPOINT_PATH = Path("checkpoints/model_h4_l6_b256_k10_it8000_lam5000.pt")
# CHECKPOINT_ITER = 8000
# print(f"loaded checkpoint - resuming training at iter {CHECKPOINT_ITER}")

# # ----------------------------
# # Hyperparameters - Small model
# # ----------------------------
# batch_size    = 16         # sequences per batch
# block_size    = 32         # context length
# max_iters     = 1000
# eval_interval = 100
# learning_rate = 1e-3
# eval_iters    = 200
# n_embd        = 64
# n_head        = 4
# n_layer       = 4
# dropout       = 0.0

# ----------------------------
# Hyperparameters - "Large" model
# ----------------------------
batch_size = 256  # sequences per batch
block_size = 256  # context length
max_iters = 40000
batch_size = 256  # sequences per batch
block_size = 256  # context length
max_iters = 40000
eval_interval = 2000
learning_rate = 1e-4
eval_iters = 250
n_embd = 256
n_head = 4
n_layer = 6
dropout = 0.1
scheduling = False  # if True, use teacher forcing with scheduled sampling
teacher_forcing_lamda = 5000  # decay rate
patience = 2


# ----------------------------
# Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Data loading helpers
# ----------------------------
def read_ids_from_txt(path: Path) -> torch.Tensor:
    """
    Reads whitespace-delimited integers from a .txt file and returns a 1D LongTensor.
    Works for either space-separated or one-id-per-line files.
    """
    s = path.read_text(encoding="utf-8")
    ids = [int(x) for x in s.split()]  # split on any whitespace
    return torch.tensor(ids, dtype=torch.long)


def read_vocab_json(path: Path) -> list[str]:
    """
    Reads a JSON list of strings. Do NOT strip spaces; tokens may include leading/trailing spaces.
    """
    vocab_tokens = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(vocab_tokens, list) or not all(
        isinstance(t, str) for t in vocab_tokens
    ):
        raise ValueError("Vocab file must be a JSON list of strings.")
    return vocab_tokens


def build_model_path(
    out_root: Path,
    n_head: int,
    n_layer: int,
    block_size: int,
    step: int,
    k: int | None = None,
    lamda: float | None = None,
):
    """
    Returns checkpoints/<folder>/model_h{n_head}_l{n_layer}_b{block_size}[_it{step}].pt
    """
    out_root.mkdir(parents=True, exist_ok=True)
    name = f"model_h{n_head}_l{n_layer}_b{block_size}_k{k}"
    if step is not None:
        name += f"_it{step}"
    if lamda is not None:
        name += f"_lam{int(lamda)}"
    return out_root / f"{name}.pt"


# ----------------------------
# Batching / eval helpers
# ----------------------------
def get_batch(
    split: str, train_data: torch.Tensor, val_data: torch.Tensor, device: str
):
    """
    Generates a batch of input (x) and target (y) sequences.
    """
    data = train_data if split == "train" else val_data
    assert (
        len(data) > block_size + 1
    ), "Dataset is too short for the configured block_size."
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_batch_with_scheduling(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: str,
    prob: float,
    model,
):
    """
    Generates a batch of input (x) and target (y) sequences. Uses teacher forcing with the probability "prob".
    """
    assert 0 <= prob <= 1, "Probability must be between 0 and 1."
    data = train_data if split == "train" else val_data
    assert (
        len(data) > block_size + 1
    ), "Dataset is too short for the configured block_size."
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x_gt = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    x = x_gt.clone()
    for t in range(
        1, block_size
    ):  # use first token as context and then use scheduled sampling / anneal teacher forcing
        use_model_pred = torch.rand(batch_size, device=device) > prob
        if use_model_pred.any():
            # No teacher forcing for at least one sample in batch: use model's prediction
            logits, _ = model(x_gt[:, :t])
            pred_token = logits[:, -1, :].argmax(dim=-1)
            # replace gt input with model prediction
            x[use_model_pred, t] = pred_token[use_model_pred]

    return x, y


@torch.no_grad()
def estimate_loss(model, train_data: torch.Tensor, val_data: torch.Tensor, device: str):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ----------------------------
# Teacher forcing / scheduled sampling helpers
# ----------------------------
def teacher_forcing_prob_exponential(iter: int, lamda: float):
    """Exponential decay of teacher forcing probability"""
    return float(torch.exp(torch.tensor(-iter / lamda)))


# ----------------------------
# Model (GPT-style decoder-only Transformer)
# ----------------------------
class Head(nn.Module):
    """One head of masked self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # scale by head size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple MLP."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # use gaussian error linear units
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # (optional) weight tying can help on small models
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, logits.size(-1))
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    torch.manual_seed(420)
    # print device
    print(f"[info] using device: {device}")

    # Load data
    train_data = read_ids_from_txt(TRAIN_IDS_TXT)
    val_data = read_ids_from_txt(VAL_IDS_TXT)

    max_train = int(train_data.max()) if train_data.numel() else -1
    max_val = int(val_data.max()) if val_data.numel() else -1
    max_id = max(max_train, max_val)

    # Load vocab (optional)
    if VOCAB_TOKENS_TXT is not None and Path(VOCAB_TOKENS_TXT).exists():
        id_to_token = read_vocab_json(VOCAB_TOKENS_TXT)
        vocab_size = len(id_to_token)
    else:
        id_to_token = None
        vocab_size = max_id + 1

    print(f"[info] max_id={max_id}, vocab_size={vocab_size}")

    # Sanity check: ensure all IDs fit in the vocab
    if max_id >= vocab_size:
        raise ValueError(
            f"Found token id >= vocab_size. "
            f"max_id={max_id} (train={max_train}, val={max_val}) >= vocab_size={vocab_size}. "
            "Load the exact vocab used to encode (same k & ordering), and make sure it's parsed as JSON."
        )

    # Build model/optimizer
    model = GPT(vocab_size).to(device)
    print(
        f"[info] model parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f}M"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # specs for saving
    specs = dict(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
    )
    out_root = Path("checkpoints")

    # tensorboard
    run_id = build_model_path(
        out_root,
        n_head,
        n_layer,
        block_size,
        step=" ",
        k=K,
        lamda=teacher_forcing_lamda,
    )
    run_id = run_id.stem[6:]  # chop off the "model_"
    writer = SummaryWriter(log_dir=f"gpt_runs/run_{run_id}")

    # Early stopping variables
    steps_without_validation_improvement = 0
    best_valid_loss = float("inf")
    best_model_state = None
    best_iter = 0

    # NOTE: to use this, uncomment checkpoint path checking at beginning of file
    # # Load from checkpoint (optional) - NOTE this will "lose" patience, so it might train a bit extra
    # if CHECKPOINT_PATH.exists():
    #     model = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #     start_iter = CHECKPOINT_ITER
    # else:
    #     start_iter = 0

    start_iter = 0
    # Train
    for iter in range(start_iter, max_iters):
        # evaluate occasionally
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            writer.add_scalar("Loss/train", losses["train"], iter)
            writer.add_scalar("Loss/val", losses["val"], iter)
            writer.add_scalar(
                "Teacher Forcing Prob.",
                teacher_forcing_prob_exponential(iter, teacher_forcing_lamda),
                iter,
            )

            # Early stopping / patience check
            if losses["val"] < best_valid_loss:
                best_valid_loss = losses["val"]
                steps_without_validation_improvement = 0
                best_model_state = model.state_dict()
                best_iter = iter
            else:
                steps_without_validation_improvement += 1

            if steps_without_validation_improvement >= patience:
                print(
                    f"Early stopping triggered at step {iter}, reverting to best model from step {best_iter}."
                )
                model.load_state_dict(best_model_state)
                break


        # Save model checkpoints regularly
        if (iter % SAVE_INTERVAL == 0 or iter == max_iters - 1) and iter > 0:
            save_path = build_model_path(
                step=iter,
                k=K,
                lamda=teacher_forcing_lamda,
                out_root=out_root,
                n_head=n_head,
                n_layer=n_layer,
                block_size=block_size,
            )
            torch.save(model, save_path)  # <-- save the entire model object
            print(f"[saved] {save_path}")

        # Update teacher annealing
        if scheduling:
            prob = teacher_forcing_prob_exponential(iter, teacher_forcing_lamda)

            xb, yb = get_batch_with_scheduling(
                "train", train_data, val_data, device, prob, model
            )
        else:
            xb, yb = get_batch("train", train_data, val_data, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    writer.close()

    # Generate simple demo at end of training
    start_ctx = torch.zeros(
        (1, 1), dtype=torch.long, device=device
    )  # assumes 0 is a valid token
    gen_ids = model.generate(start_ctx, max_new_tokens=200, temperature=1.0, top_k=50)[
        0
    ].tolist()

    if id_to_token is not None:
        out_text = "".join(id_to_token[i] for i in gen_ids if 0 <= i < len(id_to_token))
        print(out_text)
    else:
        print("Generated IDs:", gen_ids)


if __name__ == "__main__":
    main()
