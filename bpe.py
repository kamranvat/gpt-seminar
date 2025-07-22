"""
    task: look for a few different k's
	    - observe when we reach the sweet spot of subword len = ca. word len
	        (- actually not the real sweet spot cause you lose compositionality already here)
	- normalize
	- make a randomly selected test set and report the difference between test set and another test set (ie some wikipedia article or sth).
        - visualize this somehow
    - chars that can not be matched at all are counted as fail
- split 99/1 - norm (diff strats) - compare
- BPE pseudocode:
- args: corpus (as one file), nr of merges k; returns vocab
	- dict V <- all unique chars in C (write own fct)
	- do k times:
		- t_L, t_R <- most freq pair of adj tokens in c
		- t_NEW <- t_L + t_R
		- V <- V + t_NEW
		- replace each occurrence of t_L + t_R in C with t_NEW
"""
from collections import Counter
from itertools import combinations

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

def preprocess_corpus(corpus, lowercase=True):
    """Take the raw corpus and return the preprocessed corpus"""
    # if lowercase: TODO make lowercase
    # TODO split into words, return as list of strings
    pass

def get_unique_chars(corpus):
    """Get unique characters from the corpus (corpus as one str)."""
    return set(corpus)

def split_corpus(corpus):
    """Return the corpus as list of strings to prepare for further processing"""
    return list(corpus)

def get_most_frequent_pair(corpus):
    # go over all tokens, return the most frequent pair
	d  = Counter()

	if len(corpus) < 2:
		return None # TODO think about what we need to return here
     
	for comb in combinations(corpus,2):
		d[comb] += 1

	pair = d.most_common(1)[0][0]
	return pair

def get_all_pair_counts(corpus):
    # just for looking into stuff. 
    d  = Counter()
    
    for comb in combinations(corpus,2):
        d[comb] += 1

    return d.most_common()
    
def replace_most_frequent_pair(corpus, t_lr, t_l, t_r):
    # replace instances of l r with lr
    pass

def bpe(corpus, k):
    vocab = get_unique_chars(corpus)
    corpus_list = split_corpus(corpus)
    
    for i in k:
        t_l, t_r = get_most_frequent_pair(corpus_list)
        t_new = t_l + t_r
        vocab.append(t_new)
        corpus_list = replace_most_frequent_pair(corpus_list, t_new, t_l, t_r) # NOTE we could compute t_new in here
    return vocab

def main():
    # test corpus loading
    corpus_filepath = "./shakespeare.txt"
    corpus = load_corpus(corpus_filepath, 1000)
    print(corpus)
    
	# test unique chars
    uniq = get_unique_chars(corpus)
    print(uniq)
    
	# test most frequent pairs
    corpus_list = split_corpus(corpus)
    most_freq = get_most_frequent_pair(corpus_list)
    print(most_freq)
    
if __name__ == "__main__":
    main()
