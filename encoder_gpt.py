import os
import numpy as np
from pathlib import Path
from utils import FileUtils, Paths
from bpe_class import BPE

# ---- params ----
K = 5000  # <-- pick your k here

# ---- paths (constructed, not hard-coded attributes) ----
vocab_path = Path(Paths.vocab_dir) / f"vocab_nNone_k{K}.txt"
segmented_path_train = Path(Paths.tokenized_dir) / f"Shakespeare_clean_train_nNone_k{K}.txt"
segmented_path_valid = Path(Paths.tokenized_dir) / f"Shakespeare_clean_valid_nNone_k{K}.txt"

# sanity checks
for p in [vocab_path, segmented_path_train, segmented_path_valid]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing file: {p}")

# ---- BPE + encode ----
bpe = BPE()
bpe.set_vocab(FileUtils().load_vocab(vocab_path))  # same vocab used to segment

seg_tokens_train = FileUtils().load_vocab(segmented_path_train)  # list[str]
seg_tokens_valid = FileUtils().load_vocab(segmented_path_valid)  # list[str]

train_ids = bpe.encode(seg_tokens_train)
val_ids   = bpe.encode(seg_tokens_valid)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# ---- export (.bin and .txt) ----
data_dir = Path("gpt_bin")
data_dir.mkdir(parents=True, exist_ok=True)

vocab_size = len(bpe.vocab)
dtype = np.uint16 if vocab_size < (1 << 16) else np.uint32

np.array(train_ids, dtype=dtype).tofile(str(data_dir / "train.bin"))
np.array(val_ids,   dtype=dtype).tofile(str(data_dir / "val.bin"))

with open(data_dir / "train.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, train_ids)))
with open(data_dir / "val.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, val_ids)))
