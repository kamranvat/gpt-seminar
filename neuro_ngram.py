import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import ast
from itertools import product
from utils import Paths

from bpe import (
    load_corpus,
    preprocess_corpus,
    create_str_to_int_map,
    create_int_to_str_map,
    encode,
    decode,
)


class NeuroNgram(nn.Module):
    def __init__(self, vocab, n=2):
        super().__init__()
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size*self.vocab_size, self.vocab_size)
    
    def forward(self, context, target=None):
        # expected one hot encoded target
        logits = self.embedding(context)  # shape: (batch, time, vocab)

        # reshape as required for loss
        batch_size, context_size, vocab_size = logits.shape
        logits_view = logits.view(batch_size*context_size, vocab_size)
        targets_view = target.view(batch_size*context_size)

        loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss

    def predict(self, context, max_tokens=50):

        pass


    def get_batch(self, data, batch_size, context_size):
        start_indices = torch.randint(low=0, high=(len(data)-(context_size+self.n)), size=(batch_size,))
        context_start_indices = [data[j:j+context_size+self.n-2] for j in start_indices]
        c = torch.tensor(context_start_indices).unfold(1, 2, 1)
        # need to multiply first value with vocab size and then sum along last axis
        for n in range(0, self.n-2):
            c[:,:,n] *= self.vocab_size

        # sum up
        x = torch.sum(c, dim=-1)
        y = torch.stack([torch.tensor(data[i+self.n-1:i+self.n+context_size-1]) for i in start_indices])
        context = [data[j:j+context_size+self.n] for j in start_indices]
        return x, y


def test():
    # load  corpus
    corpus_path = Paths.shakespeare_clean_train
    corpus = load_corpus(corpus_path, window_size=1000)
    corpus = preprocess_corpus(corpus)
    # create model

    vocab_path = Paths.vocab_full_k250
    with open(vocab_path, "r") as f:
        list_string = f.read()
        vocab = ast.literal_eval(list_string)

    # corpus = corpus.lower()
    # print("vocab", vocab)

    string_to_int = create_str_to_int_map(vocab)
    int_to_string = create_int_to_str_map(vocab)
    print("string to int", string_to_int)
    encoded_vocab = encode(vocab, string_to_int)
    print(f"vocab size {len(vocab)}, encoded shape {len(encoded_vocab)}")
    print("vocab shape", torch.tensor(encoded_vocab).shape)
    m = NeuroNgram(vocab=torch.tensor(encoded_vocab), n=3)

    corpus = encode(corpus[:1000], string_to_int)
    print("corpus :10", corpus[:10])

    # test batching
    x, y = m.get_batch(corpus, 8, context_size=6)
    print(x.shape, y.shape)

    l, loss = m(x, y)
    print(l.shape)
    print("loss", loss)



test()
