from collections import Counter
from itertools import filterfalse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json
from utils import FileUtils, Paths


class BPE:
    def __init__(self, k=20, testset_ratio=0.1):
        """Initialize BPE class with parameters"""
        self.k = k
        self.testset_ratio = testset_ratio
        self.vocab = []
        self.str_to_int_map = {}
        self.int_to_str_map = {}

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.create_str_to_int_map(vocab)
        self.create_int_to_str_map(vocab)

    def load_vocab(self, filepath):
        fu = FileUtils()
        self.set_vocab(fu.load_vocab(filepath))

    def create_str_to_int_map(self, vocab):
        """take the vocab and return a dict mapping tokens to indices"""
        self.str_to_int_map = {token: i for i, token in enumerate(vocab)}
        return self.str_to_int_map

    def create_int_to_str_map(self, vocab):
        """take the vocab and return a dict mapping indices to tokens"""
        self.int_to_str_map = {i: token for i, token in enumerate(vocab)}
        return self.int_to_str_map

    def encode(self, tokens):
        """take list of str tokens and return list of int tokens"""
        if not self.str_to_int_map:
            if not self.vocab:
                raise ValueError("vocab is empty, cannot encode tokens.")
            self.create_str_to_int_map(self.vocab)
        return [self.str_to_int_map[token] for token in tokens]

    def decode(self, tokens):
        """take list of int tokens and return list of str tokens"""
        if not self.int_to_str_map:
            if not self.vocab:
                raise ValueError("vocab is empty, cannot decode tokens.")
            self.create_int_to_str_map(self.vocab)
        return [self.int_to_str_map[token] for token in tokens]

    def get_unique_chars(self, corpus):
        """Get unique characters from the corpus (corpus as one str)."""
        return set(corpus)

    def get_most_frequent_pair(self, corpus):
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

    def get_all_pair_counts(self, corpus):
        """Return the counts of all pairs of neighboring tokens in corpus."""
        # just for looking into stuff.
        d = Counter()
        for comb in zip(corpus, corpus[1:]):
            d[comb] += 1
        if not d:
            return None, None

        return d.most_common()

    def replace_most_frequent_pair(self, corpus, t_lr, t_l, t_r):
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

    def train(self, corpus, k):
        start = time.time()
        vocab = list(self.get_unique_chars(corpus))
        corpus_list = list(corpus)

        for i in range(0, k):
            t_l, t_r = self.get_most_frequent_pair(corpus_list)
            if t_l == None:
                print(
                    f"[WARNING] Stopped merging at k = {i} - no more pairs available!"
                )
                break
            t_new = t_l + t_r
            vocab.append(t_new)
            corpus_list = self.replace_most_frequent_pair(corpus_list, t_new, t_l, t_r)
        end = time.time()
        timer = end - start
        print(timer)
        self.vocab = vocab
        return corpus_list, vocab

    def test(self, vocab, test_set, min_token_length=3):
        """Take a vocab and a test set (as str), run bpe, return information about the performance"""
        test_set = list(test_set)  # list of str
        valid_indices = list(range(0, len(test_set)))
        matched_indices = np.zeros_like(test_set, dtype=bool)

        for token in vocab:
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

    def evaluate(self, vocab, test_set, max_n=3):
        # check percentage of text covered by all, and then with increasing n
        # all tokens of length >n
        coverages = []
        matched_chars = []
        for n in range(1, max_n + 1):
            t, coverage, m = self.test(vocab, test_set, min_token_length=n)
            coverages.append(coverage)
            matched_chars.append(m)

        return coverages

    def plot_coverages(self, vocab, train_set, test_set, max_n=3):
        coverages = self.evaluate(vocab, train_set, max_n=max_n)
        x = np.arange(start=1, stop=max_n + 1)
        test_coverages = self.evaluate(vocab, test_set, max_n=max_n)

        fig, ax = plt.subplots(figsize=(6, 2), layout="tight")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
        ax.set_xlabel("x")
        ax.plot(x, coverages, label="train")
        ax.plot(x, test_coverages, label="test")
        plt.xlabel("n")
        plt.ylabel("percentage covered")
        plt.legend()
        plt.show()

    def evaluate_token_length(self, vocab, train_set, test_set):
        """Compare metrics of the segmentation between the train and test set"""
        train_set_segmented = self.test(vocab, train_set)
        test_set_segmented = self.test(vocab, test_set)

        train_lengths = [len(token) for token in train_set_segmented]
        test_lengths = [len(token) for token in test_set_segmented]

        plt.figure(figsize=(12, 6))
        plt.hist(train_lengths, bins=30, alpha=0.5, label="Train Set")
        plt.hist(test_lengths, bins=30, alpha=0.5, label="Test Set")
        plt.axvline(
            np.mean(train_lengths), color="blue", linestyle="dashed", linewidth=1
        )
        plt.axvline(
            np.mean(test_lengths), color="orange", linestyle="dashed", linewidth=1
        )
        plt.legend()
        plt.title("Token Length Distribution")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.show()

        return None


def main():
    # paths
    shakespeare_train_path = Paths.shakespeare_clean_train
    vocab_path = Paths.vocab_full_k250
    vocab_dir_path = Paths.vocab_dir

    # params
    k = 20
    n_chars = 1000  # set to None to load full corpus
    testset_ratio = 0.1  # how much of the full corpus to use as test

    bpe_model = BPE(k=k, testset_ratio=testset_ratio)

    corpus = FileUtils().load_corpus(shakespeare_train_path, window_size=n_chars)
    test_set = FileUtils().extract_test_set(corpus, testset_ratio)
    # test_set = FileUtils().load_corpus(sms_path, window_size=n_chars)

    # test bpe
    bpe = BPE(k=k, testset_ratio=testset_ratio)
    bpe.set_vocab(FileUtils().load_vocab(vocab_path))
    tokenized_corpus_list, vocab = bpe.train(corpus=corpus, k=k)
    print(vocab)

    # store vocab
    vocab_name = f"vocab_n{n_chars}_k{k}.txt"
    FileUtils.store_vocab(vocab, vocab_dir_path, vocab_name)

    # plots
    bpe.plot_coverages(vocab, corpus, test_set, 20)
    # bpe.evaluate_token_length(vocab, corpus, test_set)


if __name__ == "__main__":
    main()
