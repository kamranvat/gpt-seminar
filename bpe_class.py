import logging
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from loading_utils import FileUtils, Paths

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class BPE:
    def __init__(self, k=20):
        """Initialize BPE class with parameters"""
        self.k = k
        self.vocab = []
        self.str_to_int_map = {}
        self.int_to_str_map = {}
        self.num_workers = os.cpu_count()
        self.pool = multiprocessing.Pool(self.num_workers)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

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
            logger.info("creating string to int mapping")
            self.create_str_to_int_map(self.vocab)
        return [self.str_to_int_map[token] for token in tokens]

    def decode(self, tokens):
        """take list of int tokens and return list of str tokens"""
        if not self.int_to_str_map:
            if not self.vocab:
                raise ValueError("vocab is empty, cannot decode tokens.")
            logger.info("creating int to string mapping")
            self.create_int_to_str_map(self.vocab)
        return [self.int_to_str_map[token] for token in tokens]
    
    def tokenize(self, text, vocab=None):
        if vocab is None:
            if not self.vocab:
                raise ValueError("you must provide a vocab list or BPE vocab must have been set")
            vocab = self.vocab

        tokenized, _, _ = self.test(vocab, text, min_token_length=0)
        return tokenized
        

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
            return None, None
        pair = d.most_common(1)[0][0]
        return pair

    def get_all_pair_counts(self, corpus):
        """Return the counts of all pairs of neighboring tokens in corpus."""
        # just for looking into stuff.
        d = Counter()
        if len(corpus) == 0:
            return d
        for comb in zip(corpus, corpus[1:]):
            d[comb] += 1

        return d

    def get_most_frequent_pair_parallelized(self, corpus):
        # split the corpus and then get the counts on multiple workers
        section_length = len(corpus)//(self.num_workers-2)
        corpus_splits = [corpus[max(n*section_length-1, 0):min(
            (n+1)*section_length, len(corpus)+1)] for n in range(self.num_workers)]

        results = self.pool.map(self.get_all_pair_counts, corpus_splits)
        counter = Counter()
        for c in results:
            counter += c

        pair = counter.most_common(1)[0][0]
        return pair

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

        for i in tqdm(range(0, k), desc="Training"):
            t_l, t_r = self.get_most_frequent_pair_parallelized(corpus_list)
            if t_l == None:
                logger.warning(
                    f"[WARNING] Stopped merging at k = {i} - no more pairs available!"
                )
                break
            t_new = t_l + t_r
            vocab.append(t_new)
            corpus_list = self.replace_most_frequent_pair(
                corpus_list, t_new, t_l, t_r)
        end = time.time()
        timer = end - start
        logger.info(f"training took {timer} s")
        self.vocab = vocab
        return corpus_list, vocab

    def test(self, vocab, test_set, min_token_length=1, tqdm_position=None):
        """Take a vocab and a test set (as str), run bpe, return information about the performance"""
        test_set = list(test_set)  # list of str
        valid_indices = list(range(0, len(test_set)))
        matched_indices = np.zeros_like(test_set, dtype=bool)

        for token in tqdm(vocab, desc="Testing", position=tqdm_position):
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

    def _test_bpe_worker(self, args):
        """Worker function for multiprocessing in evaluate function"""
        vocab, test_set, n, position = args
        t, coverage, m = self.test(
            vocab, test_set, min_token_length=n, tqdm_position=position
        )
        return t, coverage, m, position

    def evaluate(self, vocab, test_set, max_n=3):
        # check percentage of text covered by all, and then with increasing n
        # all tokens of length >n
        coverages = []
        matched_chars = []

        # arguments for worker processes
        args_list = [(vocab, test_set, n, i)
                     for i, n in enumerate(range(1, max_n + 1))]

        # multiprocessing with min(cpu_cores, max_n) workers
        num_workers = min(multiprocessing.cpu_count(), max_n)
        logger.info(f"Using {num_workers} worker(s)...")

        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(self._test_bpe_worker, args_list)

        for _, coverage, m, __ in results:
            coverages.append(coverage)
            matched_chars.append(m)

        return coverages

    def plot_coverages(self, vocabs, ks, train_set, id_test_set, ood_test_set, max_n=3, save=False, save_dir="results", save_name="coverages.png"):
        if len(vocabs) != len(ks):
            raise ValueError("vocabs and ks must have the same length")

        coverages_list = []
        id_test_coverages_list = []
        ood_test_coverages_list = []

        for vocab, k in zip(vocabs, ks):
            coverages = self.evaluate(vocab, train_set, max_n=max_n)
            id_test_coverages = self.evaluate(vocab, id_test_set, max_n=max_n)

            ood_test_coverages = self.evaluate(
                vocab, ood_test_set, max_n=max_n)
            coverages_list.extend(coverages)
            id_test_coverages_list.extend(id_test_coverages)
            ood_test_coverages_list.extend(ood_test_coverages)

        fig, ax = plt.subplots(figsize=(12, 6), layout="tight")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())

        linestyles = ['-', "--", ":"]
        x = np.tile(np.arange(start=1, stop=max_n + 1), len(ks*3))
        k_array = np.tile(np.array(ks).repeat(max_n), 3)
        train_test = ['train']*max_n * \
            len(ks) + ['ID test']*max_n*len(ks) + ['OOD test']*max_n*len(ks)

        coverages_list.extend(id_test_coverages_list)
        coverages_list.extend(ood_test_coverages_list)
        df = pd.DataFrame({"x": x, "k": k_array, "type": train_test,
                          "coverages": np.array(coverages_list)})

        sns.lineplot(data=df, x="x", y="coverages", hue="type", style="k")
        ax.set_xlabel("x")
        plt.xlabel("n")
        plt.ylabel("percentage covered")
        plt.legend()
        if save:
            os.makedirs(save_dir, exist_ok=True)
            save_path = Path(save_dir) / save_name
            plt.savefig(save_path)
        else:
            plt.show()

    def evaluate_token_length(self, vocab, train_set, id_test_set, ood_test_set, save=False, save_dir="results", save_name="token_length_distribution.png", n_chars=None, k=-1):
        """Compare metrics of the segmentation between the train and test set"""

        # arguments for worker processes
        args_list = [(vocab, train_set, 1, 0),
                     (vocab, id_test_set, 1, 1), (vocab, ood_test_set, 1, 2)]
        logger.info(f"Using {3} worker(s)...")

        with multiprocessing.Pool(3) as pool:
            results = pool.map(self._test_bpe_worker, args_list)

        for t, coverage, m, p in results:
            if p == 0:
                train_set_segmented = t
            elif p == 1:
                id_test_set_segmented = t
                # store tokenized corpus for later use
                corpus_name = f"Shakespeare_clean_test_n{n_chars}_k{k}.txt"
                os.makedirs(Paths.tokenized_dir, exist_ok=True)
                FileUtils.store_vocab(list(t), Paths.tokenized_dir, corpus_name)
            elif p == 2:
                ood_test_set_segmented = t
                # store tokenized corpus for later use
                corpus_name = f"sms_clean_test_n{n_chars}_k{k}.txt"
                os.makedirs(Paths.tokenized_dir, exist_ok=True)
                FileUtils.store_vocab(list(t), Paths.tokenized_dir, corpus_name)

        train_lengths = [len(token) for token in train_set_segmented]
        id_test_lengths = [len(token) for token in id_test_set_segmented]
        ood_test_lengths = [len(token) for token in ood_test_set_segmented]

        plt.figure(figsize=(12, 6))
        plt.hist(train_lengths, density=True, bins=30,
                 alpha=0.5, label="Train Set")
        plt.hist(id_test_lengths, density=True, bins=30,
                 alpha=0.5, label="ID Test Set")
        plt.hist(ood_test_lengths, density=True, bins=30,
                 alpha=0.5, label="OOD Test Set")
        plt.axvline(
            np.mean(train_lengths), color="blue", linestyle="dashed", linewidth=1
        )
        plt.axvline(
            np.mean(id_test_lengths), color="orange", linestyle="dashed", linewidth=1
        )
        plt.axvline(
            np.mean(ood_test_lengths), color="green", linestyle="dashed", linewidth=1
        )
        plt.legend()
        plt.title("Token Length Distribution")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        if save:
            os.makedirs(save_dir, exist_ok=True)
            save_path = Path(save_dir) / save_name
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    # paths
    shakespeare_train_path = Paths.shakespeare_clean_train
    shakespeare_test_path = Paths.shakespeare_clean_test
    sms_path = Paths.sms_clean
    vocab_path = Paths.vocab_full_k250
    vocab_dir_path = Paths.vocab_dir

    results_dir = "results_bpe"
    os.makedirs(results_dir, exist_ok=True)

    ks = [8, 12]
    n_chars = None  # set to None to load full corpus
    vocabs = []

    for k in ks:
        corpus = FileUtils().load_corpus(shakespeare_train_path, window_size=n_chars)
        id_test_set = FileUtils().load_corpus(shakespeare_test_path, window_size=n_chars)
        ood_test_set = FileUtils().load_corpus(sms_path, window_size=n_chars)

        # test bpe
        bpe = BPE(k=k)
        #bpe.set_vocab(FileUtils().load_vocab(vocab_path))
        tokenized_corpus_list, vocab = bpe.train(corpus=corpus, k=k)
        vocabs.append(vocab)

        # store vocab
        vocab_name = f"vocab_n{n_chars}_k{k}.txt"
        FileUtils.store_vocab(list(vocab), vocab_dir_path, vocab_name)

        # store tokenized corpus for later use
        corpus_name = f"Shakespeare_clean_train_n{n_chars}_k{k}.txt"
        os.makedirs(Paths.tokenized_dir, exist_ok=True)
        FileUtils.store_vocab(list(tokenized_corpus_list), Paths.tokenized_dir, corpus_name)

        #bpe.evaluate_token_length(vocab, corpus, id_test_set, ood_test_set, save=True,
        #                          save_dir=results_dir, save_name=f"token_length_distribution_k_{k}.png", n_chars=n_chars, k=k)

    # plots
    # bpe.plot_coverages(vocabs, ks, corpus, id_test_set,
    #                    ood_test_set, 10, save=True, save_dir=results_dir)


if __name__ == "__main__":
    main()
