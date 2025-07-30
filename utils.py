import json
import random


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
        filepath = filepath + name
        with open(filepath, "w") as f:
            json.dump(vocab, f)

    @staticmethod
    def load_vocab(filepath):
        """Load vocab from file"""
        with open(filepath, "r") as f:
            vocab = json.load(f)
        return vocab

    def extract_test_set(self, corpus, percentage):
        """Return randomly sampled test set of size percentage*wordcount as str"""
        split_corpus = corpus.split()
        n_words = int(percentage * len(split_corpus))
        split_corpus = random.sample(split_corpus, n_words)
        return " ".join(split_corpus)
