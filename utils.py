import json


def load_corpus(filepath, window_size=None):
    """Load corpus from filepath, return as string. If window size is passed, return subset of that size."""
    corpus = ""
    if window_size:
        with open(filepath, "r") as f:
            corpus = f.read(window_size)
    else:
        with open(filepath, "r") as f:
            corpus = f.read()
    return corpus


def store_vocab(vocab, filepath, name):
    """Store generated vocab"""
    filepath = filepath + name
    with open(filepath, "w") as f:
        json.dump(vocab, f)


def load_vocab(filepath):
    """Load vocab from file"""
    with open(filepath, "r") as f:
        vocab = json.load(f)
    return vocab
