import numpy as np
from pathlib import Path
from loading_utils import FileUtils, Paths
from bpe_class import BPE

class GPTEncoder:
    def __init__(self, k=2000):
        self.K = k
        self.vocab_path = Path(Paths.vocab_dir) / f"vocab_nNone_k{self.K}.txt"
        # Use load_tokenized from neuro_ngram.py to handle segmented files
        self.vocab = FileUtils().load_vocab(self.vocab_path)
        self.bpe = BPE(k=self.K)
        self.bpe.set_vocab(self.vocab)
        self.segmented_train = FileUtils().load_tokenized(
            'Shakespeare_clean', 'train', vocab=self.vocab, bpe=self.bpe, k=self.K, n_chars=None)
        self.segmented_valid = FileUtils().load_tokenized(
            'Shakespeare_clean', 'valid', vocab=self.vocab, bpe=self.bpe, k=self.K, n_chars=None)
        self.vocab_size = len(self.bpe.vocab)
        self.dtype = np.uint16 if self.vocab_size < (1 << 16) else np.uint32
    def _maybe_generate_segmented_files(self):
        # If segmented files do not exist, generate them using BPE
        for split in ["train", "valid"]:
            segmented_path = Path(Paths.tokenized_dir) / f"Shakespeare_clean_{split}_nNone_k{self.K}.txt"
            if not segmented_path.exists():
                # Load vocab
                vocab = FileUtils().load_vocab(self.vocab_path)
                bpe = BPE(k=self.K)
                bpe.set_vocab(vocab)
                # Load corpus
                corpus_path = Path(Paths.corpus_dir) / f"Shakespeare_clean_{split}.txt"
                corpus = FileUtils().load_corpus(corpus_path, window_size=None)
                # Tokenize and store
                tokenized_corpus, _, _ = bpe.test(vocab, corpus)
                FileUtils.store_vocab(list(tokenized_corpus), Paths.tokenized_dir, f"Shakespeare_clean_{split}_nNone_k{self.K}.txt")

    def _sanity_check_files(self):
        # Only check for vocab file, segmented files are handled by load_tokenized
        if not Path(self.vocab_path).exists():
            raise FileNotFoundError(f"Missing file: {self.vocab_path}")

    def encode(self):
        train_ids = self.bpe.encode(self.segmented_train)
        val_ids = self.bpe.encode(self.segmented_valid)
        return train_ids, val_ids

    def export(self, data_dir="gpt_bin"):
        train_ids, val_ids = self.encode()
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.array(train_ids, dtype=self.dtype).tofile(str(data_dir / "train.bin"))
        np.array(val_ids, dtype=self.dtype).tofile(str(data_dir / "val.bin"))
        with open(data_dir / "train.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, train_ids)))
        with open(data_dir / "val.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, val_ids)))
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

    def encode_string(self, text: str) -> list[int]:
        # segment with BPE
        tokens = self.bpe.tokenize(text)
        ids = self.bpe.encode(tokens)
        return ids
