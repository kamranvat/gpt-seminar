"""
    task: look for a few different k's
            - observe when we reach the sweet spot of subword len = ca. word len
                (- actually not the real sweet spot cause you lose compositionality already here)
        - normalize
        - make a randomly selected test set and report the difference between test set and another test set (ie some wikipedia article or sth).
        - visualize this somehow
    - chars that can not be matched at all are counted as fail
- split 99/1 - norm (diff strats) - compare
"""

from collections import Counter
from itertools import filterfalse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json
from tqdm import tqdm


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


def preprocess_corpus(corpus, lowercase=True, rm_whitespace=True):
    """Take the raw corpus and return the preprocessed corpus"""
    # TODO: add regex things in here
    if rm_whitespace:
        corpus = " ".join(corpus.split())
    if lowercase:
        corpus = (
            corpus.casefold()
        )  # casefold instead of lower for better handling of weird chars
    return corpus


def get_unique_chars(corpus):
    """Get unique characters from the corpus (corpus as one str)."""
    return set(corpus)


def split_corpus(corpus):  # TODO rename or replace
    """Return the corpus as list of strings to prepare for further processing"""
    return list(corpus)


def get_most_frequent_pair(corpus):
    """Return the most frequent pair of neighboring tokens in corpus"""
    d = Counter()
    if len(corpus) < 2:
        return None, None
    for comb in zip(corpus, corpus[1:]):
        d[comb] += 1
    if not d:
        None, None
    pair = d.most_common(1)[0][0]
    return pair


def get_all_pair_counts(corpus):
    """Return the counts of all pairs of neighboring tokens in corpus."""
    # just for looking into stuff.
    d = Counter()
    for comb in zip(corpus, corpus[1:]):
        d[comb] += 1
    if not d:
        return None, None

    return d.most_common()


def replace_most_frequent_pair(corpus, t_lr, t_l, t_r):
    """In corpus, replace instances of "l", "r" with "lr" """
    # note that for corpora with only one char: this might add typos of ggg -> gggg if t_lr == gg
    new_corpus = []
    skip = False
    for i in range(0, len(corpus) - 1):
        if skip:
            skip = False
        else:
            if corpus[i] == t_l and corpus[i + 1] == t_r:
                new_corpus.append(t_lr)
                skip = True
            else:
                new_corpus.append(corpus[i])
    return new_corpus


def extract_test_set(corpus, percentage):
    """Return randomly sampled test set of size percentage*wordcount as str"""
    split_corpus = corpus.split()
    n_words = int(percentage * len(split_corpus))
    split_corpus = random.sample(split_corpus, n_words)
    return " ".join(split_corpus)


def bpe(corpus, k):
    start = time.time()
    vocab = list(get_unique_chars(corpus))
    corpus_list = split_corpus(corpus)

    for i in tqdm(range(0, k), desc="Training"):
        t_l, t_r = get_most_frequent_pair(corpus_list)
        if t_l == None:
            print(f"[WARNING] Stopped merging at k = {i} - no more pairs available!")
            break
        t_new = t_l + t_r
        vocab.append(t_new)
        corpus_list = replace_most_frequent_pair(corpus_list, t_new, t_l, t_r)
    end = time.time()
    timer = end - start
    print(timer)
    return corpus_list, vocab


def test_bpe(vocab, test_set, min_token_length=3):
    """Take a vocab and a test set (as str), run bpe, return information about the performance"""
    test_set = split_corpus(test_set)  # list of str
    valid_indices = list(range(0, len(test_set)))
    matched_indices = np.zeros_like(test_set, dtype=bool)

    for token in tqdm(vocab, desc="Testing"):
        i = 0
        while i < len(valid_indices) - 1:
            l = valid_indices[i]
            r = valid_indices[i + 1]
            t_l = test_set[l]
            t_r = test_set[r]
            # match single character tokens
            if token == t_l and len(token) >= min_token_length:
                matched_indices[l] = True

            if token == t_l + t_r:
                test_set[l] = token
                del valid_indices[i + 1]
                if len(token) >= min_token_length:
                    matched_indices[l] = True
                    matched_indices[r] = True
            i += 1

    percentage_matched = np.sum(matched_indices) / len(test_set)

    return (
        np.array(test_set)[valid_indices],
        percentage_matched,
        np.sum(matched_indices),
    )


def evaluate(vocab, test_set, max_n=3):
    # check percentage of text covered by all, and then with increasing n
    # all tokens of length >n
    coverages = []
    matched_chars = []
    for n in range(1, max_n + 1):
        t, coverage, m = test_bpe(vocab, test_set, min_token_length=n)
        coverages.append(coverage)
        matched_chars.append(m)

    return coverages


def plot_coverages(vocab, train_set, test_set, max_n=3):
    coverages = evaluate(vocab, train_set, max_n=max_n)
    x = np.arange(start=1, stop=max_n + 1)
    test_coverages = evaluate(vocab, test_set, max_n=max_n)

    fig, ax = plt.subplots(figsize=(6, 2), layout="tight")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
    ax.set_xlabel("x")
    ax.plot(x, coverages, label="train")
    ax.plot(x, test_coverages, label="test")
    plt.xlabel("n")
    plt.ylabel("percentage covered")
    plt.legend()
    plt.show()


def evaluate_token_length(vocab, train_set, test_set):
    """Compare metrics of the segmentation between the train and test set"""
    train_set_segmented = test_bpe(vocab, train_set)
    test_set_segmented = test_bpe(vocab, test_set)

    train_lengths = [len(token) for token in train_set_segmented]
    test_lengths = [len(token) for token in test_set_segmented]

    plt.figure(figsize=(12, 6))
    plt.hist(train_lengths, bins=30, alpha=0.5, label="Train Set")
    plt.hist(test_lengths, bins=30, alpha=0.5, label="Test Set")
    plt.axvline(np.mean(train_lengths), color="blue", linestyle="dashed", linewidth=1)
    plt.axvline(np.mean(test_lengths), color="orange", linestyle="dashed", linewidth=1)
    plt.legend()
    plt.title("Token Length Distribution")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.show()

    return None


def main():
    # paths
    shakespeare_unclean_path = "./corpora/shakespeare.txt"
    shakespeare_clean_path = "./corpora/Shakespeare_clean_full.txt"
    shakespeare_train_path = "./corpora/Shakespeare_clean_train.txt"
    shakespeare_test_path = "./corpora/Shakespeare_clean_test.txt"
    sms_path = "./corpora/sms_clean.txt"
    vocab_dir_path = "./data/"

    # params
    k = 1500
    n_chars = None  # set to None to load full corpus
    testset_ratio = 0.1  # how much of the full corpus to use as test

    corpus = load_corpus(shakespeare_train_path, n_chars)
    corpus = preprocess_corpus(corpus)
    test_set = extract_test_set(corpus, testset_ratio)
    # test_set = load_corpus(sms_path, n_chars)

    # test bpe
    tokenized_corpus_list, vocab = bpe(corpus, k)
    # print(vocab)

    # store vocab
    if n_chars:
        vocab_name = f"vocab_n{n_chars}_k{k}.txt"
    else:
        vocab_name = f"vocab_full_k{k}.txt"
    store_vocab(vocab, vocab_dir_path, vocab_name)

    # plots
    # plot_coverages(vocab, corpus, test_set, 20)
    # evaluate_token_length(vocab, corpus, test_set)


if __name__ == "__main__":
    main()
