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

def preprocess_corpus(corpus, lowercase=True, rm_whitespace=True):
    """Take the raw corpus and return the preprocessed corpus"""
    # TODO: add regex things in here
    if rm_whitespace:
        corpus = ' '.join(corpus.split())
    if lowercase:
        corpus = corpus.casefold() # casefold instead of lower for better handling of weird chars
    return corpus

def get_unique_chars(corpus):
    """Get unique characters from the corpus (corpus as one str)."""
    return set(corpus)

def split_corpus(corpus): # TODO rename or replace
    """Return the corpus as list of strings to prepare for further processing"""
    return list(corpus)

def get_most_frequent_pair(corpus):
    """Return the most frequent pair of neighboring tokens in corpus"""
    d  = Counter()
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
    d  = Counter()
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
    for i in range(0, len(corpus)-1):    
        if skip:
            skip = False
        else:
            if corpus[i] == t_l and corpus[i+1] == t_r:
                new_corpus.append(t_lr)
                skip = True
            else:
                new_corpus.append(corpus[i])
    return new_corpus

def extract_test_set(corpus, percentage):
    """Return randomly sampled test set of size percentage*wordcount as str"""
    split_corpus = corpus.split()
    n_words = int(percentage*len(split_corpus))
    split_corpus = random.sample(split_corpus, n_words)
    return " ".join(split_corpus)

def bpe(corpus, k):
    vocab = list(get_unique_chars(corpus))
    corpus_list = split_corpus(corpus)
    
    for i in range(0, k):
        t_l, t_r = get_most_frequent_pair(corpus_list)
        if t_l == None:
            print(f"[WARNING] Stopped merging at k = {i} - no more pairs available!")
            break
        t_new = t_l + t_r
        vocab.append(t_new)
        corpus_list = replace_most_frequent_pair(corpus_list, t_new, t_l, t_r)
    return corpus_list, vocab

def test_bpe(vocab, test_set):
    """Take a vocab and a test set (as str), run bpe, return information about the performance"""
    test_set = split_corpus(test_set) # list of str
    valid_indices = list(range(0, len(test_set)))
    
    for token in vocab:
        i = 0
        while i < len(valid_indices)-1:
            l = valid_indices[i]
            r = valid_indices[i+1]
            t_l = test_set[l]
            t_r = test_set[r]
            if token == t_l + t_r:
                test_set[l] = token
                del valid_indices[i+1]
            i += 1
    return np.array(test_set)[valid_indices]
		

    
def main():
    # test corpus loading
    corpus_filepath = "./shakespeare.txt"
    corpus = load_corpus(corpus_filepath, 1000)
    corpus = preprocess_corpus(corpus)
    corpus_list = split_corpus(corpus)
    test_set = extract_test_set(corpus, 0.1)

	# test bpe
    corpus_list, vocab = bpe(corpus_list, 130)
    
	# test test_bpe
    vocab = test_bpe(vocab, corpus)
    print(vocab)
    
if __name__ == "__main__":
    main()
