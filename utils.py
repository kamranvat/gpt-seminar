import json
import random
from pathlib import Path


class FileUtils:
    """Utility functions for file operations."""

    def load_corpus(
        self,
        filepath,
        preprocess=True,
        lowercase=True,
        rm_whitespace=True,
        window_size=None,
    ):
        """Load corpus from filepath, return as string. If window size is passed, return subset of that size."""
        corpus = ""
        filepath = str(filepath)
        if window_size:
            with open(filepath, "r") as f:
                corpus = f.read(window_size)
        else:
            with open(filepath, "r") as f:
                corpus = f.read()

        if preprocess:
            corpus = self.preprocess_corpus(
                corpus, lowercase=lowercase, rm_whitespace=rm_whitespace
            )

        return corpus

    @staticmethod
    def preprocess_corpus(corpus, lowercase=True, rm_whitespace=True):
        """Take the raw corpus and return the preprocessed corpus"""
        # TODO: add regex things in here?
        if rm_whitespace:
            corpus = " ".join(corpus.split())
        if lowercase:
            corpus = (
                corpus.casefold()
            )  # casefold instead of lower for better handling of weird chars
        return corpus

    @staticmethod
    def store_vocab(vocab, filepath, name):
        """Store generated vocab"""
        name = Path(name)
        if not name.suffix:
            name = name.with_suffix(".txt")
        filepath = filepath / name
        filepath = str(filepath)
        with open(filepath, "w") as f:
            json.dump(vocab, f)

    @staticmethod
    def load_vocab(filepath):
        """Load vocab from file"""
        filepath = str(filepath)
        with open(filepath, "r") as f:
            vocab = json.load(f)
        return vocab

    def extract_test_set(self, corpus, percentage):
        """Return randomly sampled test set of size percentage*wordcount as str"""
        split_corpus = corpus.split()
        n_words = int(percentage * len(split_corpus))
        split_corpus = random.sample(split_corpus, n_words)
        return " ".join(split_corpus)


# import paths by importing utils paths.path
class Paths:
    """A class to hold paths for various datasets and vocabularies."""

    corpus_dir = Path("./corpora/")
    shakespeare_unclean = corpus_dir / "shakespeare.txt"
    shakespeare_clean_full = corpus_dir / "Shakespeare_clean_full.txt"
    shakespeare_clean_train = corpus_dir / "Shakespeare_clean_train.txt"
    shakespeare_clean_test = corpus_dir / "Shakespeare_clean_test.txt"
    shakespeare_clean_valid = corpus_dir / "Shakespeare_clean_valid.txt"
    sms_clean = corpus_dir / "sms_clean.txt"
    sms = corpus_dir / "sms.txt"
    vocab_dir = Path("./data/")
    sms_vocab_full_k250 = vocab_dir / "sms_vocab_full_k250.txt"
    sms_vocab_full_k500 = vocab_dir / "sms_vocab_full_k500.txt"
    sms_vocab_full_k750 = vocab_dir / "sms_vocab_full_k750.txt"
    vocab_full_k250 = vocab_dir / "vocab_full_k250.txt"
    vocab_full_k500 = vocab_dir / "vocab_full_k500.txt"
    vocab_full_k750 = vocab_dir / "vocab_full_k750.txt"
    vocab_full_k1000 = vocab_dir / "vocab_full_k1000.txt"
    vocab_full_k1250 = vocab_dir / "vocab_full_k1250.txt"
    vocab_full_k1500 = vocab_dir / "vocab_full_k1500.txt"
