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
    ngram_counts = defaultdict(int)

    for i in range(len(tokenized) - n + 1):
        ngram = tuple(tokenized[i:i + n])
        ngram_counts[ngram] += 1

    return Counter(ngram_counts).most_common(10)


def main():
    # test corpus loading
    corpus_filepath = r"C:\Users\acer\Downloads\shakespeare.txt"
    corpus = load_corpus(corpus_filepath, 300000)
    text = preprocess_corpus(corpus)
    top_bigram = n_gram(text, 2)
    top_trigram = n_gram(text, 3)
    top_quadgram = n_gram(text, 4)


    print("Top 10 bigram:")
    for ngram, count in top_bigram:
        print(f"{ngram}: {count}")

    print("Top 10 trigram:")
    for ngram, count in top_trigram:
        print(f"{ngram}: {count}")
    
    print("Top 10 quadgram:")
    for ngram, count in top_quadgram:
        print(f"{ngram}: {count}")

    
if __name__ == "__main__":
    main()
