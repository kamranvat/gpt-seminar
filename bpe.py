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

def preprocess_corpus(corpus, lowercase=True, rm_whitespace=True):
    """Take the raw corpus and return the preprocessed corpus"""
    # if lowercase: TODO make lowercase
    # TODO split into words, return as list of strings
    if rm_whitespace:
        corpus = ' '.join(corpus.split())
    if lowercase:
        corpus = corpus.casefold() # casefold instead of lower for better handling of weird chars

    return corpus

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
    
    for comb in zip(corpus,corpus[1:]):
        d[comb] += 1
          
    if not d:
        None, None
        
    pair = d.most_common(1)[0][0]
    return pair

def get_all_pair_counts(corpus):
    # just for looking into stuff. 
    d  = Counter()
    
    for comb in zip(corpus, corpus[1:]):
        d[comb] += 1
        
    if not d:
        return None, None

    return d.most_common()
    
def replace_most_frequent_pair(corpus, t_lr, t_l, t_r):
    # replace instances of l r with lr
    # for corpora with only one char: this might add typos of ggg -> gggg if t_lr == gg
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

def bpe(corpus, k):
    vocab = list(get_unique_chars(corpus))
    corpus_list = split_corpus(corpus)
    
    for _ in range(0, k):
        t_l, t_r = get_most_frequent_pair(corpus_list)
        t_new = t_l + t_r
        vocab.append(t_new)
        corpus_list = replace_most_frequent_pair(corpus_list, t_new, t_l, t_r)
    return vocab

def main():
    # test corpus loading
    corpus_filepath = "./shakespeare.txt"
    corpus = load_corpus(corpus_filepath, 10000)
    corpus = preprocess_corpus(corpus)
    corpus_list = split_corpus(corpus)

	# test bpe
    vocab = bpe(corpus_list, 130)
    print(vocab)
    
if __name__ == "__main__":
    main()
