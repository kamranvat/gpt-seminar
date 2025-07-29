import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import ast
from itertools import product

from bpe import load_corpus, create_str_to_int_map, create_int_to_str_map, encode, decode

class NeuroNgram(nn.Module):
    def __init__(self, vocab, n=2):
        super().__init__()
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)

        # create context mapping
        cs = self.vocab
        for i in range(1, self.n):
            # cs = torch.cartesian_prod(cs, self.vocab)
            cs = list(product(cs, self.vocab.tolist()))
            
            print(len(cs))

        cs = torch.tensor(np.array(cs))
        print(cs.shape)
        self.context_mapping = dict(zip(cs.tolist(), np.arange(len(cs))))
        self.inverse_context_mapping = dict(zip(np.arange(len(cs)), cs.tolist()))
        print("context mapping", self.context_mapping)
        # [TODO] create context mappings for all ns
    
    def forward(self, context, target=None):
        # encode context if necessary
        context = self.encode(context)

        # expected one hot encoded target
        logits = self.embedding(context) # shape: (batch, time, vocab)

        # reshape as required for loss
        logits = torch.view()

        loss = F.cross_entropy(logits, target)
        return logits, loss

    def predict(self, context):
        pass

    def encode(self, x):
        if self.n > 1:
            return torch.tensor([self.context_mapping[i] for i in x])
        else:
            return x

    def decode(self, x):
        if self.n >1:
            return torch.tensor([self.inverse_context_mapping[i] for i in x])
        else:
            return x
            

    def get_batch(self, data, batch_size, context_size):
        start_indices = torch.randint(low=0, high=(len(data)-(context_size+self.n)), size=(batch_size,))
        context_start_indices = [data[j:j+context_size] for j in start_indices]
        c = torch.tensor(context_start_indices).unfold(1, 2, 1)
        cs = []
        x = [self.encode(i) for i in torch.flatten(c, end_dim=1).tolist()]
        print("x", x)
        x = torch.tensor(x).reshape((batch_size, context_size))
        print(x)
        y = torch.stack([torch.tensor(data[i+self.n:i+context_size+1]) for i in start_indices])
        return x, y



def test():
    # load  corpus
    corpus = load_corpus("./corpora/Shakespeare_clean_train.txt")
    # create model

    with open('./data/vocab_full_k250.txt', 'r') as f:
        list_string = f.read()
        vocab = ast.literal_eval(list_string)

    corpus = corpus.lower()
    print("vocab", vocab)

    string_to_int = create_str_to_int_map(vocab)
    int_to_string = create_int_to_str_map(vocab)
    encoded_vocab = encode(vocab, string_to_int)
    # print(f"vocab size {vocab.shape}, encoded shape {len(encoded_vocab)}")
    print(torch.tensor(encoded_vocab).shape)
    m = NeuroNgram(vocab=torch.tensor(encoded_vocab), n=3)

    corpus = encode(corpus[:1000], string_to_int)
    print(corpus[:10])

    # test batching
    x, y = m.get_batch(corpus, 4, context_size=8)
    print(x, y)

test()

