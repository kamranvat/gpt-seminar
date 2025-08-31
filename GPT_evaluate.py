from pathlib import Path
import torch
import GPT_from_scratch as g
from GPT_encode import GPTEncoder
import json

# ----------------------------
# Model and Test Configuration
# ----------------------------
MODELS = [
    Path("trained_gpt_models/model_h4_l6_b256_k5_it20000_lam5000.pt"),
    Path("trained_gpt_models/model_h4_l6_b256_k10_it20000_lam5000.pt"),
    Path("trained_gpt_models/model_h4_l6_b256_k25_it16000_lam5000.pt"),
    Path("trained_gpt_models/model_h4_l6_b256_k50_it16000_lam5000.pt"),
    Path("trained_gpt_models/model_h4_l6_b128_k10_it16000.pt")
]

TEST_PROMPTS = [
    # Character entrance
    ("Enter Cleopatra.", "Character Entrance"),
    ("Enter Brutus.", "Character Entrance"),
    ("Enter Hamlet.", "Character Entrance"),
    # Sentiment
    ("Sweet joy fills the court.", "Positive Sentiment"),
    ("Dark grief weighs heavy on my heart.", "Negative Sentiment"),
    # Famous
    ("To be, or not to be: that is ", "Famous Line"),
    ("All the world's a ", "Famous Line"),
    ("All that glitters is not ", "Famous Line"),

]

# Parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.75
TOP_K = 42
TOP_P = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Helper Functions
# ----------------------------
def extract_model_params(model_path: Path):
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

def load_vocab_tokens(vocab_path: Path | None):
    if not vocab_path:
        return None
    toks = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(toks, list) or not all(isinstance(t, str) for t in toks):
        raise ValueError("Vocab must be a JSON list of strings.")
    return toks

def _alias_classes_into_main():
    import sys, types
    m = sys.modules.get("__main__")
    if m is None or m is not sys.modules["__main__"]:
        m = types.ModuleType("__main__")
        sys.modules["__main__"] = m
    m.Head = g.Head
    m.MultiHeadAttention = g.MultiHeadAttention
    m.FeedForward = g.FeedForward
    m.Block = g.Block
    m.GPT = g.GPT

def load_full_model(model_path: Path, device: str | torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        from torch.serialization import safe_globals
        with safe_globals([
            g.Head, g.MultiHeadAttention, g.FeedForward, g.Block, g.GPT
        ]):
            model = torch.load(model_path, map_location="cpu", weights_only=True)
        model.to(device).eval()
        return model
    except Exception:
        pass
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
    if hasattr(model, "block_size"):
        block_size = int(model.block_size)
    else:
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
            probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            mask = cumprobs > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits[mask] = -float("inf")
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[0].tolist()

def decode_ids(ids: list[int], id_to_token: list[str] | None):
    if id_to_token is None:
        return f"(IDs) {ids}"
    return "".join(id_to_token[i] for i in ids if 0 <= i < len(id_to_token))

# ----------------------------
# Evaluation Loop
# ----------------------------
def main():
    output_lines = []
    for model_path in MODELS:
        params = extract_model_params(model_path)
        K = params["K"]
        BLOCK_SIZE = params.get("block_size", 128)
        VOCAB_PATH = Path(f"data/vocab_nNone_k{K}.txt")
        gpt_encoder = GPTEncoder(k=K)
        id_to_token = load_vocab_tokens(VOCAB_PATH) if VOCAB_PATH.exists() else None
        output_lines.append(f"\n==============================")
        output_lines.append(f"Testing Model: {model_path.name}")
        output_lines.append(f"Params: {params}")
        output_lines.append(f"==============================\n")
        model = load_full_model(model_path, DEVICE)
        for prompt, test_type in TEST_PROMPTS:
            start_ids = gpt_encoder.encode_string(prompt.casefold())
            gen_ids = generate_ids(
                model,
                start_ids=start_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                device=DEVICE,
            )
            output_lines.append(f"--- Test: {test_type} ---")
            output_lines.append(f"Prompt: {prompt}")
            output_lines.append("Output:")
            output_lines.append(str(decode_ids(gen_ids, id_to_token)))
            output_lines.append("-------------------------\n")
    # Print and write to file
    for line in output_lines:
        print(line)
    with open("evaluation_output.txt", "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
