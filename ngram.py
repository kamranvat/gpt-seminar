from collections import Counter
from itertools import product
import numpy as np

class NGram:
    def __init__(self, vocab, n=4, laplace_smoothing=True, interpolation=False, backoff=False, lambdas=None):
        self.n = n 
        self.n_gram_contexts = [] # list of lists/arrays of  all possile ngrams of tokens in vocab
        self.vocab = vocab # list of all tokens,  serving as second index for freq/probability matrices
        self.n_gram_probabilities = []
        self.n_gram_frequencies = []
        self.laplace_smoothing = laplace_smoothing
        self.interpolation = interpolation
        self.backoff = backoff
        self.lambdas = lambdas

        self.n_gram_contexts.append({'': 0})
        self.n_gram_frequencies.append(np.zeros((1, len(self.vocab))))
        self.n_gram_probabilities.append(np.zeros((1, len(self.vocab))))
        self.vocab_dict = dict(zip(vocab, np.arange(len(vocab))))

        if n>1:
            # bigram contexts are unigrams
            self.n_gram_contexts.append(dict(zip(vocab, np.arange(len(vocab)))))
            self.n_gram_frequencies.append(np.zeros((len(vocab), len(self.vocab))))
            self.n_gram_probabilities.append(np.zeros((len(vocab), len(self.vocab))))


        for i in range(2, n):
            # generate all possible n grams
            contexts = list(product(self.n_gram_contexts[i-1], self.vocab))
            context_dict = dict(zip(contexts, np.arange(len(contexts))))
            self.n_gram_contexts.append(context_dict)
        
            # initialize matrices
            self.n_gram_frequencies.append(np.zeros((len(self.n_gram_contexts[i]), len(self.vocab))))
            self.n_gram_probabilities.append(np.zeros((len(self.n_gram_contexts[i]), len(self.vocab))))

        print(self.n_gram_contexts)

    def get_n_gram_counts(self, train, n):
        ngram_counts = Counter(
            tuple(train[i:i + n]) 
            for i in range(len(train) - n + 1)
        )
        return ngram_counts

    def train(self, train):
        # for each n
        for n in range(self.n):
            # get all counts in the training dataset
            counts = self.get_n_gram_counts(train, n+1)
            
            # write them in the matrix
            for key, value in counts.items():
                i = n-1
                if n==0:
                    context = ['']
                    i = n

                else:
                    context = key[:-1]
                index = key[-1]
                context = self.to_tuple(context)

                self.n_gram_frequencies[n][self.n_gram_contexts[n][context]][self.vocab_dict[index]] = value

            # optional smoothing
            if self.laplace_smoothing:
                self.n_gram_frequencies[n] += 1

            # normalize to probabilities
            # get contexts 
            if n == 0:
                # normalize by rows
                self.n_gram_probabilities[n] = self.n_gram_frequencies[n]/np.sum(self.n_gram_frequencies[n], axis=1)
            else:
                self.n_gram_probabilities[n] = np.copy(self.n_gram_frequencies[n])

                # normalize by frequency of the context
                for n_context, context_i in self.n_gram_contexts[n].items():
                    if type(n_context)!=tuple:
                        token = n_context
                        context = ''
                    else:
                        token = n_context[-1]
                        context = n_context[:-1]
                    if len(context)==1:
                        context = context[0]

                    context_freq = self.n_gram_frequencies[n-1][self.n_gram_contexts[n-1][context], self.vocab_dict[token]]
          
                    context_freq = max(context_freq, np.sum(self.n_gram_frequencies[n][self.n_gram_contexts[n][n_context], :]))

                    # get the rows where the context matches
                    if context_freq == 0:
                        self.n_gram_probabilities[n][self.n_gram_contexts[n][n_context], :] = 0
                    else:
                        self.n_gram_probabilities[n][self.n_gram_contexts[n][n_context], :] /= context_freq
    
    def get_probability(self, test):
        """Calculate probability on the test dataset"""
        # for each n-gram in the test dataset, get probability and sum them up
        if self.interpolation:
            if self.lambdas is None:
                # need to find good lambdas first
                raise ValueError("When isig interpolation you need to set or optimize lambdas first")
            return self.get_interpolated_probabilities(test, self.lambdas)

        probability = 1.0

        # deal with first words that are not n-gram of the correct length
        if self.n > 1:
            try:
                for n in range(self.n):
                    context = None
                    token = None
                    try:
                        token = self.vocab_dict[test[n+1]]
                    except KeyError:
                        # token is not in the vocab, TODO
                        print("token not found")

                    try:
                        context_key = '' if n==0 else self.to_tuple(test[:n])
                        context = self.n_gram_contexts[n][context_key]
                    except KeyError:
                        # token is not in the vocab, TODO
                        print("context not found, key", context_key, "N", n)
                        
                    probability *= self.n_gram_probabilities[n][context, token]
            except IndexError:
                # index out of range, text shoter than n
                pass

        for i in range(len(test)-self.n):
            n_gram = test[i:i+self.n]
            context = self.n_gram_contexts[self.n-1][self.to_tuple(n_gram[:-1])]
            token = self.vocab_dict[n_gram[-1]]
            probability *= self.n_gram_probabilities[self.n-1][context, token]
        
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
                        token = self.vocab_dict[test[n+1]]
                    except KeyError:
                        # token is not in the vocab, TODO
                        print("token not found")

                    try:
                        context_key = '' if n==0 else self.to_tuple(test[:n])
                        context = self.n_gram_contexts[n][context_key]
                    except KeyError:
                        # token is not in the vocab, TODO
                        print("context not found")
                    p += lambdas[n]*self.n_gram_probabilities[n][context, token]
                probability *= p
            except IndexError:
                # index out of range, text shoter than n
                pass

        for i in range(len(test)-self.n):
            p = 0
            # deal with unigram first
            n_gram = test[i:i+1]
            token = self.vocab_dict[n_gram[0]]
            p += lambdas[n]*self.n_gram_probabilities[0][0, token]

            for n in range(1, self.n):
                n_gram = test[i:i+n+1]
                context = self.n_gram_contexts[n][self.to_tuple(n_gram[:-1])]
                token = self.vocab_dict[n_gram[-1]]
                p += lambdas[n]*self.n_gram_probabilities[n][context, token]
            probability *= p
        
        return probability

    def optimize_lambdas(self, validation, strategy="random", samples=10000, grid=None):
        # strategy options: random, grid(search), 
        # search for lambdas that best optimize probability of the validation set
        
        rng = np.random.default_rng()
        best_lambdas = [-1,-1,-1]
        best_probability = -1
        
        if strategy=="random":
            for i in range(samples):
                # sample lambdas
                lambdas = rng.random(self.n)
                lambdas = lambdas/np.sum(lambdas)

                probability = self.get_interpolated_probabilities(validation, lambdas)
                
                if probability > best_probability:
                    best_lambdas = lambdas
                    best_probability = probability
            
        return best_lambdas

    def to_tuple(self, t):
        return tuple(t) if len(t)>1 else t[0]

    def predict(self, context, max_length=10, method="greedy"):
        # method: greedy or sample
        # need to have trained before we can predict
        # if len(self.n_gram_probabilities) == 0:
        #     raise ValueError('You need to call train() once to train the n-gram probabilities before you can predict')
        
        # if len(context)<self.n:
        #     raise ValueError(f'You need to provide a context of at least length {self.n}')

        # given context generate until end of sequence token is generated or until max_length is reached
        next_token = None
        sequence = []
        current_context = context[-(self.n-1):] if len(context)>=self.n else context

        while next_token != '∅' and len(sequence)<max_length:
            # get all probabilities where the first words match
            n = len(current_context)
            
            probabilities = None
            try:
                context_key = '' if n==0 else self.to_tuple(current_context)
                context = self.n_gram_contexts[n][context_key]
                probabilities = self.n_gram_probabilities[n][context]
            except KeyError:
                # context not in possible combinations of tokens, this should not happen!!
                # could try back off
                # TODO
                print("context not found, context:", current_context)

            # check that there is at least one match
            if np.sum(probabilities) == 0:
                if self.backoff:
                    # [TODO] back off
                    p = 0
                    temp_n = n-1
                    temp_context = current_context[:-1]
                    while p==0 and temp_n>0:
                        temp_context_idx = self.n_gram_contexts[len(temp_context)][self.to_tuple(temp_context)]
                        temp_probabilities = self.n_gram_probabilities[len(temp_context)][temp_context_idx]
                        p = np.sum(temp_probabilities)
                        temp_context = temp_context[:-1]
                        temp_n -= 1

                probabilities = temp_probabilities
        
            best_index = np.argmax(probabilities)
            next_token = self.vocab[best_index]
            print("context", current_context, "predict", next_token)
            sequence.append(next_token)
            current_context.append(next_token)
            current_context = current_context[-(self.n-1):] if len(current_context)>=self.n else current_context
        return sequence

    def test_perplexity(self, test):
        probability = self.get_probability(test)
        return probability**(1/len(test)) 
    
def test():
    text = "this is a test sentence and this is a test text"

    vocab = np.unique(text.split())
    print(f"vocab: {vocab}")


    n_gram = NGram(vocab, n=2, laplace_smoothing=True)

    print(f"initial text: {text.split()}")
    n_gram.train(text.split())
    print("frequencies", n_gram.n_gram_frequencies)

    print()
    print(" probabilities", n_gram.n_gram_probabilities)

    print('p', n_gram.get_probability('sentence and this'.split()))
    print('perplexity', n_gram.test_perplexity('sentence and this'.split()))
    print('pi', n_gram.get_interpolated_probabilities('sentence and this'.split(), [0.5, 0.5]))

    
    print('lamdas', n_gram.optimize_lambdas('this is'.split()))
    print('predict', n_gram.predict('this'.split()))


    
if __name__ == "__main__":
    test()

