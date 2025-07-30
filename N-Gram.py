from collections import Counter
from collections import defaultdict
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


def n_gram(text, n):
    tokenized = text.split()

    ngram_counts = Counter(
        tuple(tokenized[i:i + n]) 
        for i in range(len(tokenized) - n + 1)
    )
    return ngram_counts

'''
def n_gram_prob(n_gram_data, query_tuple):

    count = n_gram_data.values()
    total = sum(n_gram_data.values())

    ngram_prob = {
        ngram: {
            'probability': count / total
        }
        for ngram, count in n_gram_data.items()
    }
    return ngram_prob
'''
    
def conditional_prob(text, n, query_tuple):
    if len(query_tuple) != n:
        raise ValueError("query_tuple length must match n")

    ngram_data = n_gram(text, n)
    prefix_data = n_gram(text, n - 1) if n > 1 else None

    joint_count = ngram_data.get(query_tuple, 0)
    
    if n == 1:
        total = sum(ngram_data.values())
        return joint_count / total if total > 0 else 0.0
    else:
        prefix = query_tuple[:-1]
        prefix_count = prefix_data.get(prefix, 0)
        return joint_count / prefix_count if prefix_count > 0 else 0.0


def main():
    # test corpus loading
    corpus_filepath = r"C:\Users\acer\Downloads\shakespeare.txt"
    corpus = load_corpus(corpus_filepath, 300000)
    text = preprocess_corpus(corpus)
    top_unigram = n_gram(text, 1)
    top_bigram = n_gram(text, 2)
    top_trigram = n_gram(text, 3)
    top_quadgram = n_gram(text, 4)

    query_tuple = ('i', 'am')
    # ngram_data = top_unigram
    print(conditional_prob(text, 2, query_tuple))

'''''
    if query_tuple in ngram_data:
        prob = ngram_data[query_tuple]['probability']
        print(f"Probability of {query_tuple}: {prob}")
    else:
        print(f"{query_tuple} not found in the n-gram data.")
'''

if __name__ == "__main__":
    main()
