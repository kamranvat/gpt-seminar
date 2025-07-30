import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import Paths
from glob import glob
from torch.utils.tensorboard import SummaryWriter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

from bpe import (
    load_corpus,
    preprocess_corpus,
    create_str_to_int_map,
    create_int_to_str_map,
    encode,
    decode,
    load_vocab
)



class NeuroNgram(nn.Module):
    def __init__(self, vocab, n=2):
        super().__init__()
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(
            self.vocab_size ** self.n, self.vocab_size
        )

    def forward(self, context, target=None):
        # expected one hot encoded target
        logits = self.embedding(context)  # shape: (batch, time, vocab)

        # reshape as required for loss
        batch_size, context_size, vocab_size = logits.shape
        logits_view = logits.view(batch_size * context_size, vocab_size)
        loss = None
        if target is not None:
            targets_view = target.view(batch_size * context_size)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def predict(self, context, max_new_tokens=50):
        for _ in range(max_new_tokens):
            # predict (fwd pass)
            logits, loss = self(context)
            # look at last timestep
            logits = logits[:, -1, :]  # logits is now [batchsize, vocab_len]
            # softmax for probs
            probs = F.softmax(logits, dim=-1)
            # sample
            next_context = torch.multinomial(probs, num_samples=1)  # [batchsize, 1]
            # append sampled to sequence
            context = torch.cat((context, next_context), dim=1)  # [batchsize, t+1]
        return context

    def get_batch(self, data, batch_size, context_size):
        start_indices = torch.randint(
            low=0, high=(len(data) - (context_size + self.n)), size=(batch_size,)
        )
        context_start_indices = [
            data[j : j + context_size + self.n - 2] for j in start_indices
        ]
        c = torch.tensor(context_start_indices).unfold(1, 2, 1)
        # need to multiply first value with vocab size and then sum along last axis
        for n in range(0, self.n - 2):
            c[:, :, n] *= self.vocab_size

        # sum up
        x = torch.sum(c, dim=-1)
        y = torch.stack(
            [
                torch.tensor(data[i + self.n - 1 : i + self.n + context_size - 1])
                for i in start_indices
            ]
        )
        context = [data[j : j + context_size + self.n] for j in start_indices]
        return x, y

    def encode(self, c):
        # expects c to be the context as tensor of shape (batch_size, context_window, n-1)
        for n in range(0, self.n - 2):
            c[:, :, n] *= self.vocab_size

        # sum up
        return torch.sum(c, dim=-1)
    
    def decode(self, c):
        return c%self.vocab_size


def train(model, data, writer, batch_size=16, context_size=8, steps=10, validation_data=None, validate_every_x=1, patience=5, model_save_dir = "models", save_top_k=None):

    steps_without_validation_improvement = 0
    best_valid_loss = torch.inf

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(steps):
        # get batch
        x, y = model.get_batch(data=data, batch_size=batch_size, context_size=context_size)

        # perform one forward step
        logits, loss = model(x, y)
        writer.add_scalar("Loss/train", loss, step)

        # optimize with loss
        # zero previous gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # run validation
        if validation_data is not None and step%validate_every_x==0:
            # get batch
            x, y = model.get_batch(data=validation_data, batch_size=batch_size, context_size=context_size)

            # perform one step
            _, loss = model(x, y)
            writer.add_scalar("Loss/valid", loss, step)

            # check whether loss improved
            if loss < best_valid_loss:
                best_valid_loss = loss    
                model.save(model.state_dict(), os.path.join(model_save_dir, f"step_{step}_loss_{loss}"))

                # optionally delete oldest model
                if save_top_k is not None:
                    # match models
                    files =  glob(os.path.join(model_save_dir, f"step_"))
                    if len(files) > save_top_k:
                        extract_steps = lambda x: int(x.split('_'))[0]
                        file_steps = file_steps = [extract_steps(f) for f in files]
                        oldest_index = np.argmin(file_steps)
                        # delete oldest model
                        logger.info(f"Max number of past model weights to keep reached ({save_top_k}), deleting oldes file: {files[oldest_index]}")
                        os.remove(files[oldest_index])


            else:
                steps_without_validation_improvement += 1
            
            if steps_without_validation_improvement >= patience:
                # early stopping
                logger.info(f"Early stopping triggered at step {step}, reverting back to step {step-patience}")
                # match path
                best_path = glob(os.path.join(model_save_dir, f"step_{step-patience}_"))[0]
                model.load_state_dict(torch.load(best_path, weights_only=True))
                break

            
def main():
    ############### define parameters ####################
    n = 3
    context_size = 6
    batch_size = 16
    patience = 5 # stop early if validation loss has not improved for this number of times
    validate_every_x = 1  # run validation every x steps
    steps = 100000
    model_save_dir = "models"

    n_chars_corpus = 1000  # None for full
    ############### parameters end #######################

    writer = SummaryWriter()

    # load corpus
    corpus_path = Paths.shakespeare_clean_train
    corpus = load_corpus(corpus_path, window_size=n_chars_corpus)
    corpus = preprocess_corpus(corpus)
  
    vocab_path = Paths.vocab_full_k250
    vocab = load_vocab(vocab_path)

    string_to_int = create_str_to_int_map(vocab)
    int_to_string = create_int_to_str_map(vocab)
    encoded_vocab = encode(vocab, string_to_int)
    m = NeuroNgram(vocab=torch.tensor(encoded_vocab), n=n)

    corpus = encode(corpus[:1000], string_to_int)
    logger.debug("corpus len", len(corpus))

    # test batching
    x, y = m.get_batch(corpus, batch_size=batch_size, context_size=context_size)
    logger.debug(x.shape, y.shape)

    l, loss = m(x, y)
    logger.debug(l.shape)
    logger.debug("loss", loss)

    context = torch.unsqueeze(torch.unsqueeze(torch.tensor(corpus[:2]), 0), 0)

    context = m.encode(context)

    o = m.predict(context)
    # need to decode each sequence in batch separately
    logger.debug(decode(m.decode(o)[0].tolist(), int_to_string))

    # train model
    train(
        m, 
        data=corpus, 
        writer=writer, 
        batch_size=batch_size, 
        context_size=context_size, 
        steps=steps,
        # validation_data=
        validate_every_x=validate_every_x,
        patience=patience,
        model_save_dir=model_save_dir,
        )

    o = m.predict(context)
    # need to decode each sequence in batch separately
    logger.debug(decode(m.decode(o)[0].tolist(), int_to_string))


main()
