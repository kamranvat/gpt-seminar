from pathlib import Path
import json
import sys
import types
import torch
import torch.nn.functional as F
import GPT_from_scratch as g
from GPT_encode import GPTEncoder

# ----------------------------
# ----------------------------
MODEL_PATH = Path("checkpoints/model_h4_l6_b128_k10_it16000.pt")  # Example filename
START_TEXT = "All the world's a "

def extract_model_params(model_path: Path):
    """
    Extracts hyperparameters from the model filename.
    Expected format: model_h{n_head}_l{n_layer}_b{block_size}_k{K}_it{iter}.pt
    Returns a dict of params.
    """
    filename = model_path.stem
    parts = filename.split("_")
    params = {}
    for part in parts:
        if part.startswith("h"):
            params["n_head"] = int(part[1:])
        elif part.startswith("lam"):
            params["teacher_forcing_lamda"] = int(part[3:])
        elif part.startswith("l"):
            params["n_layer"] = int(part[1:])
        elif part.startswith("b"):
            params["block_size"] = int(part[1:])
        elif part.startswith("k"):
            params["K"] = int(part[1:])
        elif part.startswith("it"):
            params["iter"] = int(part[2:])
    return params


params = extract_model_params(MODEL_PATH)
K = params["K"]
BLOCK_SIZE = params.get("block_size", 128)  # fallback default

VOCAB_PATH = Path(f"data/vocab_nNone_k{K}.txt")
gpt_encoder = GPTEncoder(k=K)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_IDS = gpt_encoder.encode_string(START_TEXT.casefold())
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.95

SAVE_WEIGHTS_ONLY_COPY = False

# ----------------------------
# Helpers
# ----------------------------


def load_vocab_tokens(vocab_path: Path | None):
    if not vocab_path:
        return None
    toks = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(toks, list) or not all(isinstance(t, str) for t in toks):
        raise ValueError("Vocab must be a JSON list of strings.")
    return toks


def _alias_classes_into_main():
    """
    If the original checkpoint recorded classes under __main__.ClassName,
    alias the current class definitions (from module `g`) into __main__ so pickle can resolve them.
    """
    m = sys.modules.get("__main__")
    if m is None or m is not sys.modules["__main__"]:
        m = types.ModuleType("__main__")
        sys.modules["__main__"] = m
    # Expose the expected class names
    m.Head = g.Head
    m.MultiHeadAttention = g.MultiHeadAttention
    m.FeedForward = g.FeedForward
    m.Block = g.Block
    m.GPT = g.GPT


def load_full_model(model_path: Path, device: str | torch.device):
    """
    Robust loader for a full-object checkpoint saved via torch.save(model, ...),
    compatible with PyTorch 2.6 'weights_only' default and __main__ pickling.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1) Try safe load with allowlisted classes (keeps weights_only=True)
    try:
        from torch.serialization import safe_globals

        with safe_globals(
            [g.Head, g.MultiHeadAttention, g.FeedForward, g.Block, g.GPT]
        ):
            model = torch.load(model_path, map_location="cpu", weights_only=True)
        model.to(device).eval()
        return model
    except Exception:
        pass  # fall through to aliasing strategy

    # 2) Alias classes into __main__ and load with weights_only=False (trusted files only!)
    _alias_classes_into_main()
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate_ids(
    model,
    start_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str | torch.device = "cpu",
):
    model.eval()
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # infer block size
    if hasattr(model, "block_size"):
        block_size = int(model.block_size)
    else:
        # fallback: use position embedding table size
        block_size = int(model.position_embedding_table.num_embeddings)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            mask = cumprobs > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits[mask] = -float("inf")
            # map back to original index order
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)  # (B,1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[0].tolist()


def decode_ids(ids: list[int], id_to_token: list[str] | None):
    if id_to_token is None:
        return f"(IDs) {ids}"
    # Many BPE tokens include spaces; join directly
    return "".join(id_to_token[i] for i in ids if 0 <= i < len(id_to_token))


# ----------------------------
# main
# ----------------------------
def main():
    print(f"[loading] {MODEL_PATH}")
    model = load_full_model(MODEL_PATH, DEVICE)

    # (optional) create a weights-only copy for future painless loads
    if SAVE_WEIGHTS_ONLY_COPY:
        weights_path = MODEL_PATH.with_suffix(".weights.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"[saved weights-only] {weights_path}")

    id_to_token = load_vocab_tokens(VOCAB_PATH) if VOCAB_PATH else None

    gen_ids = generate_ids(
        model,
        start_ids=START_IDS,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        device=DEVICE,
    )

    if id_to_token is None:
        print("Generated IDs:", gen_ids)
    else:
        print("\n--- Generated Text ---\n")
        print(decode_ids(gen_ids, id_to_token))
        print("\n----------------------")


if __name__ == "__main__":
    main()
