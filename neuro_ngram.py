import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import Paths
from glob import glob
from torch.utils.tensorboard import SummaryWriter
import logging
from bpe_class import BPE
from utils import FileUtils
from torcheval.metrics.text import Perplexity

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class NeuroNgram(nn.Module):
    def __init__(self, vocab, n=2):
        super().__init__()
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size ** (self.n - 1), self.vocab_size)

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
        return c % self.vocab_size


def train(
    model,
    data,
    writer,
    optimizer=None,
    batch_size=16,
    context_size=8,
    steps=10,
    validation_data=None,
    validate_every_x=1,
    patience=5,
    model_save_dir=Paths.model_dir,
    save_top_k=None,
):

    steps_without_validation_improvement = 0
    best_valid_loss = torch.inf

    model_save_dir.mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    for step in range(steps):
        logger.debug(f"step {step}")
        # get batch
        x, y = model.get_batch(
            data=data, batch_size=batch_size, context_size=context_size
        )

        # perform one forward step
        logits, loss = model(x, y)
        writer.add_scalar("Loss/train", loss, step)

        # optimize with loss
        # zero previous gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # run validation
        if validation_data is not None and step % validate_every_x == 0:
            # get batch
            x, y = model.get_batch(
                data=validation_data, batch_size=batch_size, context_size=context_size
            )

            # perform one step
            _, loss = model(x, y)
            writer.add_scalar("Loss/valid", loss, step)

            # check whether loss improved
            if loss < best_valid_loss:
                best_valid_loss = loss
                torch.save(
                    model.state_dict(),
                    model_save_dir / f"step_{step}_loss_{loss}",
                )

                # optionally delete oldest model
                if save_top_k is not None:
                    # match models
                    files = glob(str(model_save_dir / f"step_*"))
                    print(files)
                    if len(files) > save_top_k:
                        extract_steps = lambda x: int(x.split("_")[2])
                        file_steps = [extract_steps(f) for f in files]
                        oldest_index = np.argmin(file_steps)
                        # delete oldest model
                        logger.info(
                            f"Max number of past model weights to keep reached ({save_top_k}), deleting oldes file: {files[oldest_index]}"
                        )
                        os.remove(files[oldest_index])

            else:
                steps_without_validation_improvement += 1

            if steps_without_validation_improvement >= patience:
                # early stopping
                logger.info(
                    f"Early stopping triggered at step {step}, reverting back to step {step-patience}"
                )
                # match path
                best_path = glob(str(model_save_dir / f"step_{step-patience}_*"))[0]
                model.load_state_dict(torch.load(best_path, weights_only=True))
                break


def evaluate(test_set):
    metric = Perplexity()
    x, y = None
    metric.update(x, y)
    perplexity = metric.compute()
    logger.info(f"perplexity {perplexity}")
    return perplexity


def generate_example(m, bpe, corpus):
    context = torch.unsqueeze(torch.unsqueeze(torch.tensor(corpus[:2]), 0), 0)
    context = m.encode(context)
    o = m.predict(context)
    # need to decode each sequence in batch separately
    logger.info(f"generated example: {''.join(bpe.decode(m.decode(o)[0].tolist()))}")


def main():
    ############### define parameters ####################
    n = 3
    context_size = 6
    batch_size = 4
    patience = (
        5  # stop early if validation loss has not improved for this number of times
    )
    validate_every_x = 1  # run validation every x steps
    steps = 3
    model_save_dir = Paths.model_dir
    vocab_dir = Paths.vocab_dir
    # set device to whats available (cuda, mps, cpu)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    ############### parameters end #######################

    # load corpus
    file_utils = FileUtils()
    train_corpus = file_utils.load_corpus(Paths.shakespeare_clean_train)
    test_corpus = file_utils.load_corpus(Paths.shakespeare_clean_test)
    valid_corpus = file_utils.load_corpus(Paths.shakespeare_clean_valid)

    # test different ks
    # ks = [250, 500, 750, 1000, 1250, 1500]
    ks = [500]

    # optimizer hyperparameters
    optimizer_hyperparameters = [
        {},
        # {'momentum': []},
        {"learning_rate": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]},
        {"learning_rate": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]},
    ]

    for k in ks:
        logger.info(f"Starting run for k {k}")
        for i in range(12):
            logger.info(f"Starting run for optimizer {i}")
            writer = SummaryWriter(comment=f"k_{k}_i_{i}")
            bpe = BPE(k=k)

            vocab_path = vocab_dir / f"vocab_full_k{k}.txt"
            bpe.load_vocab(vocab_path)
            vocab = bpe.vocab
            logger.info("loaded vocab")
            encoded_vocab = bpe.encode(vocab)
            logger.info("encoded vocab")
            encoded_train = bpe.encode(train_corpus)
            logger.info("encoded train")
            encoded_test = bpe.encode(test_corpus)
            logger.info("encoded test")
            encoded_valid = bpe.encode(valid_corpus)
            logger.info("encoded valid")

            m = NeuroNgram(vocab=torch.tensor(encoded_vocab), n=n)
            logger.info("created model")
            optimizers = [
                torch.optim.SGD(m.parameters()),
                torch.optim.SGD(m.parameters(), momentum=0.9),
                torch.optim.Adam(m.parameters(), lr=1e-6),
                torch.optim.Adam(m.parameters(), lr=1e-5),
                torch.optim.Adam(m.parameters(), lr=1e-4),
                torch.optim.Adam(m.parameters(), lr=1e-3),
                torch.optim.Adam(m.parameters(), lr=1e-2),
                torch.optim.AdamW(m.parameters(), lr=1e-6),
                torch.optim.AdamW(m.parameters(), lr=1e-5),
                torch.optim.AdamW(m.parameters(), lr=1e-4),
                torch.optim.AdamW(m.parameters(), lr=1e-3),
                torch.optim.AdamW(m.parameters(), lr=1e-2),
            ]  # could also add , torch.optim.RMSprop

            logger.info("created optimizers")
            optimizer = optimizers[i]

            generate_example(m, bpe, encoded_valid)

            # train model
            train(
                m,
                data=encoded_train,
                writer=writer,
                optimizer=optimizer,
                batch_size=batch_size,
                context_size=context_size,
                steps=steps,
                validation_data=encoded_valid,
                validate_every_x=validate_every_x,
                patience=patience,
                model_save_dir=model_save_dir / f"{k}_{i}",
                save_top_k=1,
            )

            generate_example(m, bpe, encoded_valid)


main()
