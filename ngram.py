import logging
import multiprocessing
import os
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from bpe_class import BPE
from tqdm import tqdm
from loading_utils import FileUtils, Paths

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class NGram:
    def __init__(
        self,
        vocab,
        n=4,
        laplace_smoothing=True,
        interpolation=False,
        backoff=False,
        lambdas=None,
    ):
        self.n = n
        # list of lists/arrays of  all possile ngrams of tokens in vocab
        self.n_gram_contexts = []
        # list of all tokens,  serving as second index for freq/probability matrices
        self.vocab = vocab
        self.n_gram_probabilities = []
        self.n_gram_frequencies = []
        self.laplace_smoothing = laplace_smoothing
        self.interpolation = interpolation
        self.backoff = backoff
        self.lambdas = lambdas

        self.n_gram_contexts.append({"": 0})
        self.n_gram_frequencies.append(np.zeros((1, len(self.vocab))))
        self.n_gram_probabilities.append(np.zeros((1, len(self.vocab))))
        self.vocab_dict = dict(zip(vocab, np.arange(len(vocab))))

        self.num_workers = os.cpu_count()
        self.pool = multiprocessing.Pool(self.num_workers)

        # n-grams with n larger than 1 have contexts
        if n > 1:
            # bigram contexts are unigrams
            self.n_gram_contexts.append(dict(zip(vocab, np.arange(len(vocab)))))
            self.n_gram_frequencies.append(np.zeros((len(vocab), len(self.vocab))))
            self.n_gram_probabilities.append(np.zeros((len(vocab), len(self.vocab))))

        for i in range(2, n):
            # generate all possible n gram contexts for n>2
            contexts = list(product(self.n_gram_contexts[i - 1], self.vocab))
            context_dict = dict(zip(contexts, np.arange(len(contexts))))
            self.n_gram_contexts.append(context_dict)

            # initialize matrices
            self.n_gram_frequencies.append(
                np.zeros((len(self.n_gram_contexts[i]), len(self.vocab)))
            )
            self.n_gram_probabilities.append(
                np.zeros((len(self.n_gram_contexts[i]), len(self.vocab)))
            )

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        return self_dict

    def get_n_gram_counts(self, train, n):
        ngram_counts = Counter(
            tuple(train[i : i + n]) for i in range(len(train) - n + 1)
        )
        return ngram_counts

    def train(self, train):
        # for each n
        for n in tqdm(range(self.n), desc="n", position=0):
            # get all counts in the training dataset
            counts = self.get_n_gram_counts(train, n + 1)

            # write them in the matrix
            for key, value in tqdm(
                counts.items(), desc="counts", position=1, leave=False
            ):
                if n == 0:
                    context = [""]

                else:
                    context = key[:-1]
                index = key[-1]
                context = self.to_tuple(context)

                self.n_gram_frequencies[n][self.n_gram_contexts[n][context]][
                    self.vocab_dict[index]
                ] = value

            # optional smoothing
            if self.laplace_smoothing:
                self.n_gram_frequencies[n] += 1

            # normalize to probabilities
            # get contexts
            if n == 0:
                # normalize by rows
                self.n_gram_probabilities[n] = self.n_gram_frequencies[n] / np.sum(
                    self.n_gram_frequencies[n], axis=1
                )
            else:
                self.n_gram_probabilities[n] = np.copy(self.n_gram_frequencies[n])

                # normalize by frequency of the context
                for n_context, context_i in tqdm(
                    self.n_gram_contexts[n].items(),
                    desc="items",
                    position=2,
                    leave=False,
                ):
                    if type(n_context) != tuple:
                        token = n_context
                        context = ""
                    else:
                        token = n_context[-1]
                        context = n_context[:-1]
                    if len(context) == 1:
                        context = context[0]

                    context_freq = self.n_gram_frequencies[n - 1][
                        self.n_gram_contexts[n - 1][context], self.vocab_dict[token]
                    ]

                    context_freq = max(
                        context_freq,
                        np.sum(
                            self.n_gram_frequencies[n][
                                self.n_gram_contexts[n][n_context], :
                            ]
                        ),
                    )

                    # get the rows where the context matches
                    if context_freq == 0:
                        self.n_gram_probabilities[n][
                            self.n_gram_contexts[n][n_context], :
                        ] = 0
                    else:
                        self.n_gram_probabilities[n][
                            self.n_gram_contexts[n][n_context], :
                        ] /= context_freq

    def get_probability(self, test):
        """Calculate log probability on the test dataset"""
        # for each n-gram in the test dataset, get probability and sum them up
        if self.interpolation:
            if self.lambdas is None:
                # need to find good lambdas first
                raise ValueError(
                    "When usig interpolation you need to set or optimize lambdas first"
                )
            return self.get_interpolated_probabilities(test, self.lambdas)

        probability = 1.0

        # deal with first words that are not n-gram of the correct length
        if self.n > 1:
            try:
                for n in range(self.n):
                    context = None
                    token = None
                    try:
                        token = self.vocab_dict[test[n + 1]]
                    except KeyError:
                        # token is not in the vocab, skip it
                        logger.warning(f"token '{token}' not part of the vocabulary")
                        continue

                    try:
                        context_key = "" if n == 0 else self.to_tuple(test[:n])
                        context = self.n_gram_contexts[n][context_key]
                    except KeyError:
                        # token is not in the vocab, skip for nwo
                        logger.warning(f"context '{context}' not found, N {n}")
                        continue

                    probability += np.log(self.n_gram_probabilities[n][context, token])
            except IndexError:
                # index out of range, text shoter than n, we can ignore this
                pass

        for i in range(len(test) - self.n):
            n_gram = test[i : i + self.n]
            context = self.n_gram_contexts[self.n - 1][self.to_tuple(n_gram[:-1])]
            token = self.vocab_dict[n_gram[-1]]
            probability += np.log(self.n_gram_probabilities[self.n - 1][context, token])

        return probability

    def get_interpolated_probabilities(self, test, lambdas):
        probability = 1.0

        # deal with first words that are not n-gram of the correct length
        if self.n > 1:
            p = 0
            try:
                for n in range(self.n):
                    context = None
                    token = None
                    try:
                        token = self.vocab_dict[test[n + 1]]
                    except KeyError:
                        # token is not in the vocab, skip it
                        logger.warning(f"token '{token}' not part of the vocabulary")
                        continue

                    try:
                        context_key = "" if n == 0 else self.to_tuple(test[:n])
                        context = self.n_gram_contexts[n][context_key]
                    except KeyError:
                        # token is not in the vocab, skip for now
                        logger.warning(f"context '{context}' not found, N {n}")
                        continue

                    p += lambdas[n] * self.n_gram_probabilities[n][context, token]
                probability += np.log(p)
            except IndexError:
                # index out of range, text shoter than n, we can ignore this
                pass

        for i in range(len(test) - self.n):
            p = 0
            # deal with unigram first
            n_gram = test[i : i + 1]
            token = self.vocab_dict[n_gram[0]]
            p += lambdas[0] * self.n_gram_probabilities[0][0, token]

            for n in range(1, self.n):
                n_gram = test[i : i + n + 1]
                context = self.n_gram_contexts[n][self.to_tuple(n_gram[:-1])]
                token = self.vocab_dict[n_gram[-1]]
                p += lambdas[n] * self.n_gram_probabilities[n][context, token]
            probability += np.log(p)

        return probability

    def _random_optimizer_worker(self, args):
        samples, validation, tqdm_position = args
        rng = np.random.default_rng()
        best_lambdas = [-1, -1, -1]
        best_probability = -np.inf

        for _ in tqdm(range(samples), position=tqdm_position):
            # sample lambdas
            lambdas = rng.random(self.n)
            lambdas = lambdas / np.sum(lambdas)

            probability = self.get_interpolated_probabilities(validation, lambdas)

            if probability > best_probability:
                best_lambdas = lambdas
                best_probability = probability

        return best_lambdas, best_probability

    def optimize_lambdas(
        self, validation, strategy="random", samples=1000, grid=None, parallelize=True
    ):
        # strategy options: random, grid(search),
        rng = np.random.default_rng()
        best_lambdas = [-1, -1, -1]
        best_probability = -np.inf

        if strategy == "random":
            if parallelize:
                # search for lambdas that best optimize probability of the validation set
                # arguments for worker processes
                args_list = [
                    (samples // self.num_workers, validation, i)
                    for i in range(self.num_workers - 1)
                ]
                # handle last one specifically as the number of samples might be different
                args_list.append(
                    (
                        samples
                        - ((samples // self.num_workers) * (self.num_workers - 1)),
                        validation,
                        self.num_workers - 1,
                    )
                )

                results = self.pool.map(self._random_optimizer_worker, args_list)

                for lambdas, probability in results:
                    if probability > best_probability:
                        best_lambdas = lambdas
                        best_probability = probability
            else:
                for _ in tqdm(range(samples)):
                    # sample lambdas
                    lambdas = rng.random(self.n)
                    lambdas = lambdas / np.sum(lambdas)

                    probability = self.get_interpolated_probabilities(
                        validation, lambdas
                    )

                    if probability > best_probability:
                        best_lambdas = lambdas
                        best_probability = probability

        if strategy == "grid":
            # TODO: (optional) implement grid search optimization
            raise NotImplementedError(
                "Grid search oprimization of lamdas has not been implemented yet, use strategy 'random' instead"
            )

        self.lambdas = best_lambdas
        return best_lambdas, best_probability

    def to_tuple(self, t):
        return tuple(t) if len(t) > 1 else t[0]

    def predict(
        self,
        context,
        max_length=10,
        method="greedy",
        end_of_sequence_tokens=[".", "!", "?"],
    ):
        # method: greedy or sample

        # given context generate until end of sequence token is generated or until max_length is reached
        next_token = None
        sequence = []
        context = list(context)
        current_context = (
            context[-(self.n - 1) :] if len(context) >= self.n else context
        )

        while next_token not in end_of_sequence_tokens and len(sequence) < max_length:
            # get all probabilities where the first words match
            n = len(current_context)

            probabilities = None
            try:
                context_key = "" if n == 0 else self.to_tuple(current_context)
                context = self.n_gram_contexts[n][context_key]
                probabilities = self.n_gram_probabilities[n][context]
            except KeyError:
                # context not in possible combinations of tokens, this should not happen!!
                # could try back off
                # TODO
                logger.error("context not found, context:", current_context)
            except IndexError:
                if self.n == 1:
                    probabilities = self.n_gram_probabilities[self.n - 1]
                else:
                    logger.error("index error, n {self.n}, context {context}")

            # check that there is at least one match
            if np.sum(probabilities) == 0:
                logger.debug(
                    f"probabilities sum to 0, need to back off if possible - context: {context}"
                )
                if self.backoff:
                    # [TODO] back off
                    p = 0
                    temp_n = n - 1
                    temp_context = current_context[:-1]
                    while p == 0 and temp_n > 0:
                        temp_context_idx = self.n_gram_contexts[len(temp_context)][
                            self.to_tuple(temp_context)
                        ]
                        temp_probabilities = self.n_gram_probabilities[
                            len(temp_context)
                        ][temp_context_idx]
                        p = np.sum(temp_probabilities)
                        temp_context = temp_context[:-1]
                        temp_n -= 1

                    probabilities = temp_probabilities

            if method == "greedy":
                best_index = np.argmax(probabilities)
            elif method == "sample":
                probabilities = probabilities.flatten()
                if np.isclose(np.sum(probabilities), 1.0):
                    best_index = np.random.choice(
                        np.arange(len(self.vocab)), size=1, p=probabilities
                    )[0]
                else:
                    logger.debug(
                        f"probabilites do not sum up to one, but to {np.sum(probabilities)} - using uniform sampling instead"
                    )
                    best_index = np.random.choice(np.arange(len(self.vocab)), size=1)[0]
            else:
                raise ValueError("method must be either 'greedy' or 'sample'")
            next_token = self.vocab[best_index]
            sequence.append(next_token)
            current_context.append(next_token)
            current_context = (
                current_context[-(self.n - 1) :]
                if len(current_context) >= self.n
                else current_context
            )
        return sequence

    def test_perplexity(self, test):
        probability = self.get_probability(test)
        return np.power(2, -probability / len(test))


def main():
    # paths
    vocab_dir_path = Paths.vocab_dir
    results_dir = Path("final_results") / "ngram"
    csv_path = Path(results_dir) / "perplexities.csv"
    context = "All the world's a"
    test_k = 1000
    ks = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]

    use_laplace_smoothing = True
    use_interpolation = True

    lambda_samples = 10000
    max_len = 256

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"use laplace smoothing {use_laplace_smoothing}")
    logger.info(f"use interpolation {use_interpolation}")
    logger.info(f"sampled for lambda optimization: {lambda_samples}")
    logger.info(f"max len of generated examples: {max_len}")
    logger.info(f"tested ks: ks")
    logger.info(f"fixed k for ngrams: {test_k}")

    # lists for storing results
    list_n = []
    list_k = []
    list_pp = []
    list_type = []

    for k in ks:
        logger.info(f"starting bigram run for k {k}")
        # load vocab for current k
        vocab_path = vocab_dir_path / f"vocab_nNone_k{k}.txt"
        vocab = FileUtils().load_vocab(vocab_path)

        bpe = BPE(k=k)
        bpe.set_vocab(vocab)
        tokenized_context = bpe.tokenize(context)
        logger.info(f"tokenized context {tokenized_context}")

        tokenized_train = FileUtils().load_tokenized(
            "Shakespeare_clean", "train", vocab=vocab, bpe=bpe, k=k, n_chars=None
        )
        tokenized_test = FileUtils().load_tokenized(
            "Shakespeare_clean", "test", vocab=vocab, bpe=bpe, k=k, n_chars=None
        )
        tokenized_valid = FileUtils().load_tokenized(
            "Shakespeare_clean", "valid", vocab=vocab, bpe=bpe, k=k, n_chars=None
        )

        # train bigram for different ks and see how that influences perplexity
        bigram = NGram(
            vocab,
            n=2,
            laplace_smoothing=use_laplace_smoothing,
            interpolation=use_interpolation,
        )

        # generate untrained example
        logger.info(
            f"untrained generation (greedy), k={k}, n=2: {context}{''.join(bigram.predict(tokenized_context, max_length=max_len))}"
        )
        for i in range(5):
            pred = bigram.predict(
                tokenized_context, max_length=max_len, method="sample"
            )
            logger.info(
                f"{i} untrained generation (sampling), k={k}, n=2: {context}{''.join(pred)}"
            )

        s = time.time()
        bigram.train(tokenized_train)
        e = time.time()
        logger.info(f"finished training in {e-s} s")

        if bigram.interpolation:
            s = time.time()
            best_lambdas, best_probability = bigram.optimize_lambdas(
                tokenized_valid, samples=lambda_samples
            )
            e = time.time()
            logger.info(f"finished lambda optimization in {e-s} s")
            logger.info(
                f"best lambdas: {best_lambdas}, with probability: {best_probability}"
            )

        # compute perplexity on train
        pp = bigram.test_perplexity(tokenized_train)
        logger.info(f"train perplexity, k={k}, n=2: {pp}")
        list_n.append(2)
        list_k.append(k)
        list_type.append("train")
        list_pp.append(pp)
        # compute perplexity on test
        pp = bigram.test_perplexity(tokenized_test)
        logger.info(f"test perplexity, k={k}, n=2: {pp}")
        list_n.append(2)
        list_k.append(k)
        list_type.append("test")
        list_pp.append(pp)
        # compute perplexity on valid
        pp = bigram.test_perplexity(tokenized_valid)
        logger.info(f"valid perplexity, k={k}, n=2: {pp}")
        list_n.append(2)
        list_k.append(k)
        list_type.append("valid")
        list_pp.append(pp)

        # generate trained examples
        pred = bigram.predict(tokenized_context, max_length=max_len)
        logger.info(
            f"trained generation (greedy), k={k}, n=2: {context}{''.join(pred)}"
        )
        logger.info(f"tokenized prediction: {pred}")
        for i in range(5):
            pred = bigram.predict(
                tokenized_context, max_length=max_len, method="sample"
            )
            logger.info(
                f"{i} trained generation (sampling), k={k}, n=2: {context}{''.join(pred)}"
            )
            logger.info(f"{i} tokenized prediction: {pred}")

        df = pd.DataFrame(
            {"k": list_k, "n": list_n, "perplexity": list_pp, "type": list_type}
        )
        df.to_csv(csv_path)

    # train n-gram for n=1, 3, 4 with fixed k
    # load vocab for current k
    k = test_k
    vocab_path = vocab_dir_path / f"vocab_nNone_k{k}.txt"
    vocab = FileUtils().load_vocab(vocab_path)

    bpe = BPE(k=k)
    bpe.set_vocab(vocab)
    tokenized_context, _, _ = bpe.test(vocab, context)
    logger.info(f"tokenized context {tokenized_context}")

    tokenized_train = FileUtils().load_tokenized(
        "Shakespeare_clean", "train", vocab=vocab, bpe=bpe, k=k, n_chars=None
    )
    tokenized_test = FileUtils().load_tokenized(
        "Shakespeare_clean", "test", vocab=vocab, bpe=bpe, k=k, n_chars=None
    )
    tokenized_valid = FileUtils().load_tokenized(
        "Shakespeare_clean", "valid", vocab=vocab, bpe=bpe, k=k, n_chars=None
    )

    for n in [1, 3, 4]:
        # train n_gram for different ks and see how that influences perplexity
        n_gram = NGram(
            vocab,
            n=n,
            laplace_smoothing=use_laplace_smoothing,
            interpolation=use_interpolation,
        )

        # generate untrained example
        logger.info(
            f"untrained generation, k={k}, n={n}: {context}{''.join(n_gram.predict(tokenized_context, max_length=max_len))}"
        )
        for i in range(5):
            pred = n_gram.predict(
                tokenized_context, max_length=max_len, method="sample"
            )
            logger.info(
                f"{i} untrained generation (sampling), k={k}, n={n}: {context}{''.join(pred)}"
            )

        s = time.time()
        n_gram.train(tokenized_train)
        e = time.time()
        logger.info(f"finished training in {e-s} s")

        if n_gram.interpolation:
            logger.info(f"starting lambda optimization")
            s = time.time()
            # parallelization creates copies of the ngram object for each thread which is too memory intensive for larger n
            parallelize = False if n > 2 else True
            best_lambdas, best_probability = n_gram.optimize_lambdas(
                tokenized_valid, samples=lambda_samples, parallelize=parallelize
            )
            e = time.time()
            logger.info(f"finished lambda optimization in {e-s} s")
            logger.info(
                f"best lambdas: {best_lambdas}, with probability: {best_probability}"
            )

        # compute perplexity on train
        pp = n_gram.test_perplexity(tokenized_train)
        logger.info(f"train perplexity, k={k}, n={n}: {pp}")
        list_n.append(n)
        list_k.append(k)
        list_type.append("train")
        list_pp.append(pp)
        # compute perplexity on test
        pp = n_gram.test_perplexity(tokenized_test)
        logger.info(f"test perplexity, k={k}, n={n}: {pp}")
        list_n.append(n)
        list_k.append(k)
        list_type.append("test")
        list_pp.append(pp)
        # compute perplexity on valid
        pp = n_gram.test_perplexity(tokenized_valid)
        logger.info(f"valid perplexity, k={k}, n={n}: {pp}")
        list_n.append(n)
        list_k.append(k)
        list_type.append("valid")
        list_pp.append(pp)

        # generate trained examples
        pred = n_gram.predict(tokenized_context, max_length=max_len)
        logger.info(
            f"trained generation (greedy), k={k}, n={n}: {context}{''.join(pred)}"
        )
        logger.info(f"tokenized prediction: {pred}")
        for i in range(5):
            pred = n_gram.predict(
                tokenized_context, max_length=max_len, method="sample"
            )
            logger.info(
                f"{i} trained generation (sampling), k={k}, n={n}: {context}{''.join(pred)}"
            )
            logger.info(f"{i} tokenized prediction: {pred}")

        df = pd.DataFrame(
            {"k": list_k, "n": list_n, "perplexity": list_pp, "type": list_type}
        )
        df.to_csv(csv_path)


if __name__ == "__main__":
    main()
