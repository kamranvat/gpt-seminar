import logging
import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bpe_class import BPE
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loading_utils import FileUtils, Paths

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
        self.embedding = nn.Embedding(
            self.vocab_size ** (self.n - 1), self.vocab_size)

    def forward(self, context, target=None):
        # expected one hot encoded target
        logits = self.embedding(context)  # shape: (batch, time, vocab)

        # reshape as required for loss
        batch_size, context_size, vocab_size = logits.shape
        logits_view = logits.view(batch_size * context_size, vocab_size)
        loss = None
        pp = None
        if target is not None:
            targets_view = target.view(batch_size * context_size)
            probs = torch.log(F.softmax(logits_view))
            target_probs = probs[torch.arange(len(probs)), targets_view]
            pp = torch.pow(2, -torch.sum(target_probs)/(len(targets_view)))

            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss, pp

    def predict(self, context, max_new_tokens=50):
        prediction = context.detach().clone()
        prediction = prediction.squeeze(0)

        # expects context to not be encoded
        for i in range(max_new_tokens):
            # predict (fwd pass)
            if self.n == 1:
                encoded_context = torch.tensor([[0]])
            else:
                # add batch dimension back for forward pass
                context = prediction[0, -(self.n-1)
                                          :].clone().unsqueeze(0).unsqueeze(0)
                encoded_context = self.encode(context)
            logits, loss, _ = self(encoded_context)
            # look at last timestep
            logits = logits[:, -1, :]  # logits is now [batchsize, vocab_len]
            # softmax for probs
            probs = F.softmax(logits, dim=-1)
            # sample
            next_context = torch.multinomial(
                probs, num_samples=1)  # [batchsize, 1]
            # append sampled to sequence

            prediction = torch.cat(
                (prediction, next_context), dim=1)  # [batchsize, t+1]
        return prediction[0, self.n-1:]

    def get_batch(self, data, batch_size, context_size):
        start_indices = torch.randint(
            low=0, high=(len(data) - (context_size + self.n)), size=(batch_size,)
        )
        context_start_indices = [
            data[j: j + context_size + self.n - 2] for j in start_indices
        ]
        c = torch.unfold_copy(torch.tensor(
            context_start_indices), 1, self.n-1, 1)
        # need to multiply first value with vocab size and then sum along last axis
        for n in range(0, self.n - 2):
            c[:, :, n] *= self.vocab_size

        # sum up
        x = torch.sum(c, dim=-1)
        y = torch.stack(
            [
                torch.tensor(data[i + self.n - 1: i +
                             self.n + context_size - 1])
                for i in start_indices
            ]
        )
        return x, y

    def encode(self, c):
        # expects c to be the context as tensor of shape (batch_size, context_window, n-1)
        for n in range(0, self.n - 2):
            c[:, :, n] *= self.vocab_size

        # sum up
        return torch.sum(c, dim=-1, dtype=int)

    def decode(self, c):
        return c % self.vocab_size

    def evaluate_perplexity_on_test(self, tokenized_test, batch_size=None, context_size=None):
        # reshape to get batches
        # [TODO] adapt to be able to use actual batches
        # for now stick to batches of size 1 to avoid issues when it cannot be divided by the batch size
        # need to keep context size to 1 as well in that case

        context = tokenized_test[:-1]
        target = tokenized_test[self.n-1:]

        c = torch.unfold_copy(torch.tensor(
            context).unsqueeze(0), 1, self.n-1, 1)
        # need to multiply first value with vocab size and then sum along last axis
        for n in range(0, self.n - 2):
            c[:, :, n] *= self.vocab_size

        # sum up
        x = torch.sum(c, dim=-1)
        y = torch.tensor(target).unsqueeze(0)
        print("x", x)
        print("y", y)
        logits, loss, pp = self.forward(context=x, target=y)

        # calculate perplexity from logits
        return pp


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
    generate_context=None,
    generate_examples_every_x=100,
    bpe=None
):

    steps_without_validation_improvement = 0
    best_valid_loss = torch.inf

    model_save_dir.mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    for step in tqdm(range(steps)):
        logger.debug(f"step {step}")
        # get batch
        x, y = model.get_batch(
            data=data, batch_size=batch_size, context_size=context_size
        )

        # perform one forward step
        logits, loss, pp = model(x, y)
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("Perplexity/train", pp, step)

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
            _, loss, pp = model(x, y)
            writer.add_scalar("Loss/valid", loss, step)
            writer.add_scalar("Perplexity/valid", pp, step)

            # check whether loss improved
            if loss < best_valid_loss:
                best_valid_loss = loss
                steps_without_validation_improvement = 0

                # optionally delete oldest model
                if save_top_k is not None:
                    # match models
                    files = glob(str(model_save_dir / f"step_*"))

                    if len(files) > save_top_k:
                        def extract_steps(x): return int(x.split("_")[3])
                        file_steps = [extract_steps(f) for f in files]
                        oldest_index = np.argmin(file_steps)
                        # delete oldest model
                        logger.info(
                            f"Max number of past model weights to keep reached ({save_top_k}), deleting oldes file: {files[oldest_index]}"
                        )
                        os.remove(files[oldest_index])

                torch.save(
                    model.state_dict(),
                    model_save_dir / f"step_{step}_loss_{loss}",
                )

            else:
                steps_without_validation_improvement += 1

            if steps_without_validation_improvement >= patience:
                # early stopping
                logger.info(
                    f"Early stopping triggered at step {step}, reverting back to step {step-patience}"
                )
                # match path
                best_path = glob(
                    str(model_save_dir / f"step_{step-patience}_*"))[0]
                model.load_state_dict(torch.load(best_path, weights_only=True))
                break

        if generate_examples_every_x > 0 and step % generate_examples_every_x == 0:
            for i in range(5):
                generate_example(model, bpe, generate_context)


def generate_example(m, bpe, tokenized_context):
    # encode context with bpe
    context = bpe.encode(tokenized_context)

    # cut to correct length for n and convert to tesor
    context = torch.unsqueeze(torch.unsqueeze(
        torch.tensor(context[:m.n-1]), 0), 0)
    o = m.predict(context)
    # need to decode each sequence in batch separately
    logger.info(
        f"generated example: {''.join(tokenized_context)}{''.join(bpe.decode(o.tolist()))}")


def main():
    ############### define parameters ####################
    n = 3
    context_size = 128
    batch_size = 64
    patience = 100  # stop early if validation loss has not improved for this number of times
    validate_every_x = 1  # run validation every x steps
    steps = 10000
    save_top_k = 1
    generate_example_every = 500
    model_save_dir = Paths.model_dir
    vocab_dir_path = Paths.vocab_dir
    results_dir = f"results_neural_ngram_n{n}"
    csv_path = Path(results_dir) / f"perplexities_n{n}.csv"
    # Ensure results directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    context = "All the world's a stage,"

    # set device to whats available (cuda, mps, cpu)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    ############### parameters end #######################

    context = FileUtils().preprocess_corpus(context)


    os.makedirs(results_dir, exist_ok=True)

    # test different ks
    ks = [100, 250, 500, 750, 1000, 1500, 2000, 5000, 7500, 10000]

    # lists for storing results
    list_i = []
    list_n = []
    list_k = []
    list_pp = []
    list_type = []

    for k in ks:
        logger.info(f"Starting run for k {k}")
        for i in range(12):
            logger.info(f"Starting run for optimizer {i}")
            writer = SummaryWriter(comment=f"n{n}_k_{k}_i_{i}")

            vocab_path = vocab_dir_path / f"vocab_nNone_k{k}.txt"
            vocab = FileUtils().load_vocab(vocab_path)

            bpe = BPE(k=k)
            bpe.set_vocab(vocab)

            tokenized_train = FileUtils().load_tokenized(
                'Shakespeare_clean', 'train', vocab=vocab, bpe=bpe, k=k, n_chars=None)
            tokenized_test = FileUtils().load_tokenized(
                'Shakespeare_clean', 'test', vocab=vocab, bpe=bpe, k=k, n_chars=None)
            tokenized_valid = FileUtils().load_tokenized(
                'Shakespeare_clean', 'valid', vocab=vocab, bpe=bpe, k=k, n_chars=None)

            tokenized_context = bpe.tokenize(context)
            logger.info(f"tokenized context {tokenized_context}")

            logger.info("loaded vocab")
            encoded_vocab = bpe.encode(vocab)
            encoded_context = bpe.encode(tokenized_context)
            logger.info("encoded vocab")
            encoded_train = bpe.encode(tokenized_train)
            logger.info("encoded train")
            encoded_test = bpe.encode(tokenized_test)
            logger.info("encoded test")
            encoded_valid = bpe.encode(tokenized_valid)
            logger.info("encoded valid")

            m = NeuroNgram(vocab=torch.tensor(encoded_vocab), n=n)
            logger.info("created model")
            optimizers = [
                torch.optim.SGD(m.parameters()),
                torch.optim.SGD(m.parameters(), momentum=0.9),
                torch.optim.SGD(m.parameters(), momentum=0.9, lr=1e-1),
                torch.optim.SGD(m.parameters(), momentum=0.9, lr=25e-2),
                torch.optim.Adam(m.parameters(), lr=1e-2),
                torch.optim.Adam(m.parameters(), lr=1e-1),
                torch.optim.Adam(m.parameters(), lr=25e-2),
                torch.optim.Adam(m.parameters(), lr=5e-1),
                torch.optim.AdamW(m.parameters(), lr=1e-2),
                torch.optim.AdamW(m.parameters(), lr=1e-1),
                torch.optim.AdamW(m.parameters(), lr=25e-2),
                torch.optim.AdamW(m.parameters(), lr=5e-1),

            ]  # could also add , torch.optim.RMSprop

            logger.info("created optimizers")
            optimizer = optimizers[i]

            for _ in range(5):
                generate_example(m, bpe, tokenized_context)
            pp = m.evaluate_perplexity_on_test(encoded_test)
            logger.info(f"untrained perplexity: {pp.item()}")

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
                model_save_dir=model_save_dir / f"n{n}_k{k}_i{i}",
                save_top_k=save_top_k,
                generate_context=tokenized_context,
                generate_examples_every_x=generate_example_every,
                bpe=bpe
            )

            # compute perplexity on train
            pp = m.evaluate_perplexity_on_test(encoded_train)
            logger.info(f"train perplexity, k={k}, n={n}, i={i}: {pp.item()}")
            list_i.append(i)
            list_n.append(n)
            list_k.append(k)
            list_type.append("train")
            list_pp.append(pp.item())
            # compute perplexity on test
            pp = m.evaluate_perplexity_on_test(encoded_test)
            logger.info(f"test perplexity, k={k}, n={n}, i={i}: {pp.item()}")
            list_i.append(i)
            list_n.append(n)
            list_k.append(k)
            list_type.append("test")
            list_pp.append(pp.item())
            # compute perplexity on valid
            pp = m.evaluate_perplexity_on_test(encoded_valid)
            logger.info(f"valid perplexity, k={k}, n={n}, i={i}: {pp.item()}")
            list_i.append(i)
            list_n.append(n)
            list_k.append(k)
            list_type.append("valid")
            list_pp.append(pp.item())

            df = pd.DataFrame(
                {"k": list_k, "n": list_n, "i": list_i, "perplexity": list_pp, "type": list_type})
            df.to_csv(csv_path)

            logger.info("finished training")
            for _ in range(5):
                generate_example(m, bpe, tokenized_context)


def test():
    ############### define parameters ####################
    n = 1
    context_size = 6
    batch_size = 4
    patience = 5  # stop early if validation loss has not improved for this number of times
    validate_every_x = 1  # run validation every x steps
    steps = 3
    model_save_dir = "models"

    context = "All the world's a stage,"

    context = FileUtils().preprocess_corpus(context)
    ############### parameters end #######################
    k = 250

    # load corpus
    file_utils = FileUtils()
    vocab_dir_path = Paths.vocab_dir
    vocab_path = vocab_dir_path / f"vocab_nNone_k{k}.txt"
    vocab = FileUtils().load_vocab(vocab_path)

    bpe = BPE(k=k)
    bpe.set_vocab(vocab)

    tokenized_train = load_tokenized(
        'Shakespeare_clean', 'train', vocab=vocab, bpe=bpe, k=k, n_chars=None)
    tokenized_test = load_tokenized(
        'Shakespeare_clean', 'test', vocab=vocab, bpe=bpe, k=k, n_chars=None)
    tokenized_valid = load_tokenized(
        'Shakespeare_clean', 'valid', vocab=vocab, bpe=bpe, k=k, n_chars=None)

    logger.info(f"loaded vocab {len(vocab)}, {vocab}")
    encoded_vocab = bpe.encode(vocab)
    logger.info(f"encoded vocab {len(encoded_vocab)}, {encoded_vocab}")

    print("train", tokenized_train[:10])
    encoded_train = bpe.encode(tokenized_train)
    print("train encoded", encoded_train[:10])
    print("nique in encoded train", np.unique(np.array(encoded_train)))
    print("nique in encoded vocab", np.unique(np.array(encoded_vocab)))
    logger.info("encoded train")
    encoded_test = bpe.encode(tokenized_test)
    logger.info("encoded test")
    encoded_valid = bpe.encode(tokenized_valid)
    logger.info("encoded valid")
    tokenized_context = bpe.tokenize(context)
    logger.info(f"tokenized context {tokenized_context}")

    m = NeuroNgram(vocab=np.array(encoded_vocab), n=n)

    x, y = m.get_batch(data=encoded_train,
                       batch_size=batch_size, context_size=context_size)
    print(x)
    print(y)

    print(x.shape, y.shape)

    logits, loss, pp = m.forward(x, y)
    print("logits \n", logits.shape, logits)
    print("loss", loss)

    generate_example(m, bpe, tokenized_context)
    pp = m.evaluate_perplexity_on_test(encoded_test)


main()
