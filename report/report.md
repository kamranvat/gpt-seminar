# REPORT
This report describes our implementation of the four milestone tasks in the "Building GPT from Scratch" seminar, from simple statistical methods to more sophisticated neural approaches. The code is structured modularly, with separate implementations for BPE tokenization, classical n-gram models, neural n-gram networks, and GPT models, and this report describes both the functionality and the results of the respective parts of the implementation.


## Task 0: Text Analysis
To get an overview of the data, we implemented both Unix-style commands and Python equivalents for analyzing the initial Shakespeare corpus, as shown below.

### 0.1 Space-based Tokenization
The initial corpus analysis was performed using space-based tokenization to understand word frequency distributions:

Bash command:
```bash
tr -sc 'A-Za-z' '\n' < shakespeare.txt | sort | uniq -c | sort -n -r
```

Python implementation:
```python
import re
import numpy as np

def tokenize():
    with open('shakespeare.txt') as f:
        s = f.read()
        pattern = r'[^A-Za-z]+'
        n = re.sub(pattern=pattern, repl='\n', string=s)
        n = n.split('\n') 
        values, counts = np.unique(n, return_counts=True)
        sorted_indices = np.argsort(counts)
        values = values[sorted_indices]
        counts = counts[sorted_indices]
        return list(zip(reversed(counts), reversed(values)))
```
The analysis revealed the most frequent tokens in the original corpus:
```
23288 the
22225 I
18653 and
16373 to
15725 of
12797 a
12186 you
10839 my
10016 in
8954 d
```
<!-- This frequency distribution follows the expected Zipfian distribution typical of natural language, with function words dominating the vocabulary. -->
As expected, function words dominate the vocabulary.


## Task 1: Data Preparation and Segmentation with BPE

### 1.1 Data Splitting and Cleaning
The initial corpus required substantial preprocessing to remove metadata and licensing information that appeared throughout the text. Our preprocessing pipeline included:

- Whitespace normalization: Collapsing multiple whitespace characters into single spaces using `' '.join(corpus.split())`
- Case normalization: Converting to lowercase using `corpus.casefold()` for better Unicode handling
- Corpus splitting: Extracting test sets as random samples of specified percentages

The cleaning process significantly reduced corpus size:

Original corpus: 4,941,133 characters (901,325 words)
Cleaned corpus: 1,041,007 characters (190,999 words)

A cleaned version of the corpus with proper train/test/validation splits was kindly provided by course participant Mohamed Ebrahim, which we used for subsequent experiments.


### 1.2 BPE Implementation
We implemented Byte Pair Encoding (BPE) as a subword tokenization method. The algorithm follows these steps:

1. Initialization: Start with individual characters as the initial vocabulary
2. Pair counting: Count all adjacent token pairs in the corpus
3. Merging: Replace the most frequent pair with a new merged token
4. Iteration: Repeat until k merges are completed

Our implementation supports both serial and parallelized processing:
```python
def get_most_frequent_pair(self, corpus):
    """Return the most frequent pair of neighboring tokens in corpus"""
    d = Counter()
    if len(corpus) < 2:
        return None, None
    for comb in zip(corpus, corpus[1:]):
        d[comb] += 1
    if not d:
        return None, None
    pair = d.most_common(1)[0][0]
    return pair
```

### 1.3 Performance Evaluation
We evaluated BPE performance across different test sets to measure generalization:
- In-domain (ID) test: Shakespeare test set from the same source
- Out-of-domain (OOD) test: SMS dataset with different linguistic characteristics

We used two methods to evaluate performance, coverage analysis and comparison of token length distribution.
- Coverage analysis: Percentage of characters in the text that could be represented by tokens of minimum length n. This metric captures how well the BPE vocabulary generalizes to new text while avoiding over-segmentation.
    ```python
    def evaluate(self, vocab, test_set, max_n=3):
        # check percentage of text covered by all tokens, then with increasing n
        coverages = []
        matched_chars = []
        
        for n in range(1, max_n + 1):
            t, coverage, m = self.test(vocab, test_set, min_token_length=n)
            coverages.append(coverage)
            matched_chars.append(m)
        
        return coverages
    ```

- Token length distribution: compares the distribution of the token lengths of the segmented text for different datasets. The distributions should be similar if the tokenizer generalized well to the other dataset. 

With these coverage metrics, we tested multiple values of k (number of merges) ranging from 100 to 10,000.

#### Coverages
![alt coverages](../final_results/bpe/coverages.png)

The coverages for train and in-distribution test can be considered identical, but are lower for the OOD test set. Based on this it seems that the test dataset is a good representation of the data distribution of the training set. 
For the OOD test set, not even using all tokens (from length 1) leads to a full coverage, highlighting the difference in these datasets. As the OOD dataset consist of SMS spam messages, this result is not surprising, but it does show the merit of using this performance measure for evaluating the generalization capability of the trained BPE. 
With a larger number of merges (larger k) a larger percentage of the corpora is covered even by shorter tokens compared to using fewer merges when training the tokenizer. While the coverage on the OOD test set is generally lower than on the train or ID test set, the trend of decreasing coverage within creasingly larger token length (n) is similar for all datasets. 

#### Token Length Distributions
![alt text](../final_results/bpe/token_length_distribution_k_10.png)
![alt text](../final_results/bpe/token_length_distribution_k_50.png)
![alt text](../final_results/bpe/token_length_distribution_k_50.png)
![alt text](../final_results/bpe/token_length_distribution_k_100.png)
![alt text](../final_results/bpe/token_length_distribution_k_250.png)
![alt text](../final_results/bpe/token_length_distribution_k_500.png)
![alt text](../final_results/bpe/token_length_distribution_k_1000.png)
![alt text](../final_results/bpe/token_length_distribution_k_2000.png)

As in the coverage plot, there is a high similarity between the token length distributions for the train and in-distribution test set, whereas the OOD test set displays different distributions that become markedly more different with increasing k, displaying a higher proportion of shorter tokens compared to the train or ID test set. This is to be expected due to the very different nature of the text contained in the OOD test set which likely cannot be segmented very well with the longer tokens more specific to the Shakespeare corpus. 


## Task 2: N-gram Engine
> "All the world's a stage, and all the men and women merely players; they have their exits and their entrances, and one man in his time plays many parts."

 — William Shakespeare

> "_All the world's a stage,_'--or 'i' the world is slain. iras cleopatra o noble weak, lo, here it is. iago nay, but not in my heart of breath: it is a customach, adieutenant-gown upon the best to have it so. o, iago? iago why, by making quince, snout snout snout snout sn"

 — Shakespeare n-gram (k=1000, n=3, greedy sampling)

### 2.1 Data Usage and Validation Strategy
As defined in the task, we used the cleaned Shakespeare corpus with proper train/test/validation splits for all n-gram experiments. The validation set served specifically for hyperparameter optimization in interpolation models, while k values were treated as a separate experimental variable. 

### 2.2 N-gram Engine Implementation
Our n-gram implementation supports arbitrary n (provided adequate computational resources are available) and includes both Laplace smoothing and simple interpolation. The n-gram engine operates on BPE-tokenized text, allowing for subword-level language modeling. This approach combines the benefits of subword regularization with statistical language modeling. The core architecture uses matrices to store frequencies and probabilities for all n:

```python
class NGram:
    def _init_(self, vocab, n=4, laplace_smoothing=True, interpolation=False, backoff=False, lambdas=None):
        self.n = n
        self.n_gram_contexts = []  # contexts for each n-gram level
        self.vocab = vocab
        self.n_gram_probabilities = []  # probability matrices
        self.n_gram_frequencies = []   # frequency matrices
```

Laplace smoothing in the training by adding one to all frequencies:
``` python
def train(self, train):
    (...)
    # optional smoothing
        if self.laplace_smoothing:
            self.n_gram_frequencies[n] += 1
    (...)
```

Interpolated probabilities are obtained by interpolating between probabilities for a token and the corresponding context for different ns, with the degree of influence determined by lambdas. These hyperparameters have to be optimized on the validation set after the N-gram has been trained (i.e. the frequency counts have been obtained) and before interpolated probabilities can be computed. 
We implemented a simple optimization of the lambda hyperparameters using random search with optional parallelization to speed up the process of drawing many samples for optimization: 

```python
def optimize_lambdas(self, validation, strategy="random", samples=1000, grid=None, parallelize=True):
    # strategy options: random, grid(search),
    rng = np.random.default_rng()
    best_lambdas = [-1, -1, -1]
    best_probability = -np.inf

    if strategy == "random":
        if parallelize:
            # search for lambdas that best optimize probability of the validation set
            # arguments for worker processes
            args_list = [(samples//self.num_workers, validation, i)
                            for i in range(self.num_workers-1)]
            # handle last one specifically as the number of samples might be different
            args_list.append((samples-((samples//self.num_workers)
                                * (self.num_workers-1)), validation, self.num_workers-1))

            results = self.pool.map(
                self._random_optimizer_worker, args_list)

            for lambdas, probability in results:
                if probability > best_probability:
                    best_lambdas = lambdas
                    best_probability = probability
        else:
            for _ in tqdm(range(samples)):
                # sample lambdas
                lambdas = rng.random(self.n)
                lambdas = lambdas/np.sum(lambdas)

                probability = self.get_interpolated_probabilities(
                    validation, lambdas)

                if probability > best_probability:
                    best_lambdas = lambdas
                    best_probability = probability

    self.lambdas = best_lambdas
    return best_lambdas, best_probability
```

### 2.3 Intrinsic Evaluation
#### 2.3.1 Perplexity Computation
Perplexity serves as our primary intrinsic evaluation metric:
```python
def test_perplexity(self, test):
    probability = self.get_probability(test)
    return np.power(2, -probability/len(test))
```

#### 2.3.2 Multi-order Analysis
We tested n-gram orders from 1 to 4, examining how increased context length affects modeling performance. Higher-order models capture more linguistic structure but suffer from increased sparsity. <!-- sparsity? [TODO] -->
[TODO] plot and description

#### 2.3.3 Effect of k on perplexity
We systematically evaluated how different values of k (BPE merge operations) affect perplexity when n is kept constant. As specified in the task, we focused on bigrams due to their balance of context and computational efficiency.
As evident from the plot below, higher ks tend to lead to higher perplexity values, not only on the test but also on the training and validation sets. This is not surprising, however, as higher k values correspond to more merges in the BPE, and thus a larger vocabulary. This leads to individual probabilities in the matrix being reduced due to the larger number of possibilities and the smoothing being applied.
[TODO] plot


### 2.4 Extrinsic Evaluation
We implemented a method for generating text from the n-gram models to allow for extrinsic, qualitative evaluation. 

#### 2.4.1 Text Generation
Our n-gram models support both greedy and sampling-based text generation:
```python
def predict(self, context, max_length=10, method="greedy", end_of_sequence_tokens=['.', '!', '?']):
    # Generate text using either greedy selection or probabilistic sampling
    while next_token not in end_of_sequence_tokens and len(sequence) < max_length:
        if method == "greedy":
            best_index = np.argmax(probabilities)
        elif method == "sample":
            best_index = np.random.choice(np.arange(len(self.vocab)), p=probabilities)
        
        next_token = self.vocab[best_index]
        sequence.append(next_token)
```

#### 2.4.2 Context Handling
The generation system handles variable-length contexts and implements fallback strategies for unseen n-grams. End-of-sequence tokens (periods, exclamation marks, question marks) provide natural stopping criteria for generated text, but a maximum length of the generation is also enforced. 

#### 2.4.3 Results
Below are some generated examples for the different n-gram models.
All models were provided the same context – `All the world's a ` – which was preprocessed and tokenized using BPE.
**Effect of **
- n=1, k=1000


## Task 3: Neural N-Gram
### 3.1 Neural N-Gram using PyTorch
#### 3.1.1 Neural Architecture
We implemented a neural n-gram model using PyTorch, representing contexts as learned embeddings:
```python
class NeuroNgram(nn.Module):
    def _init_(self, vocab, n=2):
        super()._init_()
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size ** (self.n - 1), self.vocab_size)
```
The model encodes n-gram contexts as single indices by treating them as vocab_size**(n-1) numbers, enabling direct embedding lookup.


#### 3.1.1.2 Training Infrastructure
We implemented early stopping with configurable patience to prevent overfitting. 
The system saves top-k model checkpoints based on validation performance, automatically managing storage by removing older checkpoints when the limit is exceeded. When early stopping is triggered the model is reverted to the previously best checkpoint. 


```python
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
        _, loss, pp = model(x, y)
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
                        def extract_steps(x): return int(x.split("_")[3][1:])
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

```

#### 3.1.3 Hyperparameter Optimization
We conducted systematic hyperparameter exploration across multiple dimensions:
- Optimizer Variants: We tested 12 different optimizer configurations:
    - SGD with default parameters (without momentum and learning rate=1e-2)
    - SGD with momentum=0.9 and learning rates: 1e-2, 1e-1, 25e-2
    - Adam with learning rates: 1e-4, 1e-2, 1e-1, 25e-2, 5e-1
    - AdamW with learning rates: 1e-4, 1e-2, 1e-1, 25e-2, 5e-1

- BPE Vocabulary Size (k): Tested values from 10 to 5000 merges to study the effect of subword vocabulary size on neural model performance: 10, 25, 50, 100, 250, 500, 1000, 2000, 5000

- Architecture Parameters: 
    - batch size: 256
    - patience: 100
    - steps: 10000 (never reached due to early stopping)
    - validation was performed at every training step
    - n: 1, 2, 3


#### 3.1.4 Evaluation and Generation
The neural n-gram model supports the same evaluation metrics as classical n-grams:
```python
def evaluate_perplexity_on_test(self, tokenized_test, batch_size=None, context_size=None):
    # Compute perplexity on test data using the trained neural model
    logits, loss, pp = self.forward(context=x, target=y)
    return pp
```

Text generation uses probabilistic sampling from the learned distribution:

```python
def predict(self, context, max_new_tokens=50):
    for i in range(max_new_tokens):
        logits, loss, _ = self(encoded_context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_context = torch.multinomial(probs, num_samples=1)
        prediction = torch.cat((prediction, next_context), dim=1)
    return prediction
```

Note that due to batching generated sequences are always of the specified maximum length and do not end when an end-of-sequence token is reached. 

#### 3.1.5 Experimental Results

[TODO] look these over, make sure they are still true for the new results!!!

Our experiments revealed several key insights:

- BPE Vocabulary Size: Larger vocabularies (higher k) generally improved performance up to a point, with diminishing returns beyond k=2000-5000 depending on the model type.
N-gram Order: Classical n-grams showed better performance around n=3-4, balancing context richness with sparsity issues.

- Smoothing Effectiveness: Linear interpolation consistently outperformed simple Laplace smoothing, with optimized lambda weights providing substantial improvements.

- Neural vs. Classical: Neural n-gram models showed promise but required careful hyperparameter tuning to match classical model performance.

- Generalization: All models showed expected degradation when tested on out-of-domain data (SMS vs. Shakespeare), with BPE helping to maintain some robustness.

### 3.2 Hardcore Neural N-Gram Using Only numpy

## Task 4: GPT
> "_All the world's a_ monster. i have my bond, and i think my master of a chamber had been that endeavour and hath breath of my sword, then i have look'd to bear me thus a soldier's course of passion from the heaven, if thou wilt not go to me to be play. "

 — ShakespeareGPT (trained for 16k steps, k=10, block_size=128)

<!-- [TODO]: add the parts from the new parts -->

Our journey through natural language processing is nearing it's end and brought us to implement a full GPT-style-transformer.

### Building the Neural Architecture

Our GPT implementation centers around the self-attention mechanism, implemented through individual attention heads:
```python
class Head(nn.Module):
    """One head of masked self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```
We think of it as a more sophisticated conversation system. Each token asks questions (queries) about what it's looking for, while all previous tokens offer answers (keys) about what information they contain. The actual content that gets passed forward (values) is then weighted by how well the questions match the answers.

<!-- [TODO] masking != attention -->
The triangular mask ensures our model plays by the rules - it can't peek into the future during training, maintaining the sequential nature so that the language modeling becomes meaningful.

So why stop at one conversation? When our multi-head attention can run in parallel:

```python
class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
```

After attention determines what's relevant, feed-forward networks process this information:

```python
class FeedForward(nn.Module):
    """Simple MLP."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
```

The expansion to four times the embedding dimension provides computational space for more complex transformations, while the contraction back maintains dimensional consistency throughout the network.
Bringing it all together, each transformer block combines these components: 

```python
class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Self-attention with residual connection
        x = x + self.ff(self.ln2(x))  # Feed-forward with residual connection
        return x
```

The residual connections (x = x + ...) should avoid the vanishing gradient problem, allowing us to stack many layers without issue. They also improve learning by making the loss landscape smoother.
Our complete GPT model orchestrates all these components into a cohesive system:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Weight tying optimization
        self.lm_head.weight = self.token_embedding_table.weight
```

The architecture is beautifully logical (are we bragging? maybe. is it deserved? i don't know): token embeddings provide semantic meaning, position embeddings add sequential awareness, transformer blocks perform the heavy lifting of pattern recognition and context integration, and the final layer translates back to vocabulary predictions.

Then lets move to our Seamless Integration with our BPE foundation.
One of the satisfying aspects of our implementation is how seamlessly it integrates with the BPE tokenization system we built earlier. Our data preparation pipeline transforms Shakespeare's prose into the numerical sequences our transformer craves:

(code)
### encoder_gpt.py - The bridge between BPE and neural networks
```python
bpe = BPE()
bpe.set_vocab(FileUtils().load_vocab(vocab_path))
seg_tokens_train = FileUtils().load_vocab(segmented_path_train)
train_ids = bpe.encode(seg_tokens_train)
```

### Efficient storage for training
```python
dtype = np.uint16 if vocab_size < (1 << 16) else np.uint32
np.array(train_ids, dtype=dtype).tofile(str(data_dir / "train.bin"))
```

This integration should ensure that our transformer operates on the same meaningful subword units that proved effective in our earlier n-gram experiments. The vocabulary size optimization demonstrates attention to computational efficiency - using 16-bit integers when possible to reduce memory usage and improve training speed.

### Training the Neural Bard
Training a transformer requires a lot of computing power. Our batch processing system efficiently samples training sequences:


```python
def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor, device: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
```

This sliding window approach ensures our model sees diverse examples while maintaining the sequential relationships we need for language learning.

Small Model (64 dimensions, 4 heads, 4 layers): Perfect for experimentation and understanding the architecture. We used this for testing and debugging.
"Large" Model (256 dimensions, 4 heads, 6 layers): Our training configuration, with more performance, but still small enough to run on our equipment. This was our primary experimental setup, on which the models we evaluate below were trained.

The training loop incorporates modern best practices:

```python
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model, train_data, val_data, device)
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    
    if iter % 1000 == 0:
        save_path = build_model_path(out_root, n_head, n_layer, block_size, step=iter)
        torch.save(model, save_path)
    
    xb, yb = get_batch("train", train_data, val_data, device)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### HOW TO GENERATE
Once trained, our GPT transforms (pun intended) into a digital Shakespeare capable of generating new nonsense through sophisticated sampling strategies. The generation system offers control over creativity and coherence:

```python
def generate_ids(model, start_ids, max_new_tokens, temperature=0.7, top_k=50, top_p=0.9):
    for _ in range(max_new_tokens):
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        
        # Top-k filtering concentrates on most likely options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        
        # Nucleus sampling provides dynamic vocabulary selection
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            mask = cumprobs > top_p
            sorted_logits[mask] = -float("inf")
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat((idx, idx_next), dim=1)
```

Temperature controls the model's creativity - low values produce conservative, predictable continuations, while higher values encourage ... let's call them "bold and surprising" choices. Top-k sampling focuses attention on the most promising options, while nucleus (top-p) sampling dynamically adjusts the vocabulary size based on probability mass, providing more nuanced control over generation diversity.
    
### Results

We trained GPT models on differently tokenized corpora to compare the resulting effects. We chose the following values for k (BPE-segmentations): 

_5; 10; 25; 50_

These were chosen since for the neural n-gram, <!-- [TODO was it the neural or regular?]--> we found lower perplexity values for lower values of k. This result seems to hold, at least in GPT training: 

[TODO] add graphic of training here

We initially trained one model with a context size of 128, 4 attention heads, and 6 layers on a corpus segmented with k=10. Training was performed on a Nvidia RTX 5070ti GPU for 50.000 iterations, to observe the amount of overfitting we would achieve. The training and validation loss during training can be seen below:

This yielded three key results:
- the results at roughly the time of divergence were pretty good compared to the n-grams (see quote at the beginning of this chapter), but we felt there might still be some improvement possible. 
- the validation loss diverges strongly from the training loss, but behaves relatively smoothly. 
- the training run took about 9.5 hours, of which a significant part was the very frequent evaluation to get fine-grained observations. The best loss was achieved at around 12k steps. 
- We used about 6GB of VRAM (of about 16 available) during training.

Based on these observations, we decided to 

- implement teacher forcing annealing, since gradually moving away from corpus tokens towards model-generated tokens for the input is said to be essential in training transformer models.
- set a relatively low value of 2 for the patience parameter, as it seemed unlikely to us that the validation loss would go back down after rising twice
- reduce the evaluation frequency and increase batch size, to reduce GPU load and utilize more of our available VRAM.

Based on the reasoning above, we settled on the following parameters for training:
```python
batch_size = 256         # sequences per batch
block_size = 256         # context length
max_iters = 40000        # max. iterations (unless patience is reached)
eval_interval = 2000     # evaluation frequency
eval_iters = 250         # number of batches per evaluation
learning_rate = 1e-4     # optimizer learning rate
n_embd = 256             # embedding dimension
n_head = 4               # number of attention heads
n_layer = 6              # number of transformer layers
dropout = 0.1            # dropout rate
teacher_forcing_lamda = 5000  # teacher forcing exponential decay rate
patience = 2             # early stopping patience
```

Teacher forcing probability at iteration $i$ was computed using exponential decay:

$$
p_{\text{TF}}(i) = \exp\left(-\frac{i}{\lambda}\right)
$$

where $i$ is the current iteration and $\lambda$ is the decay rate hyperparameter. With the resulting (decaying) chance, the model would get teacher-forced input during training. 

With this setup, we trained the four models for k's 5, 10, 25, and 50, each of them reaching their optimal validation loss at 16.000 or 20.000 training steps, respectively:

[TODO] insert tensorboard plots

### Evaluation
The models reach the following perplexities, confirming the earlier (n-gram) result that a lower k leads to lower perplexity.
| BPE Merges (k) | Validation Perplexity |
|:--------------:|:--------------------:|
|       5        |        [TODO]          |
|      10        |        [TODO]          |
|      25        |        [TODO]          |
|      50        |        [TODO]          |

Since these numerical values do not say much about how the model performs, we also performed a set of qualitative tests for each model by varying the initial prompt. 
We perfomed tests of...
- ...character entrance: Prompts like "enter brutus.", which are common in the original corpus. For a good model, we expect some likeness to the original work the character is from in the resulting text, like for example the names of other characters from the same play.
- ...sentiment: By starting with a positive or negative sentiment, we tried to test whether the model could produce text with matching sentiment.
- ...famous lines: Admittedly, this was mostly for comedic effect - but we still hoped to see the models to follow this up with appropriate output.

Output was generated with the following parameters:
- Max New Tokens: `256`
- Temperature: `0.75`
- Top-k: `42`
- Top-p (nucleus sampling): `0.95`


The full set of tests for each model is stored in evaluation_output.txt, which can be found in this repository. To summarize the results:
- character entrances generally cause related names to show up in the prompt. 
At k=5: _enter cleopatra. mark antony now, i do not live to a kingdom and say you, the liar of her cursed with the discontinuance sense._
- sentiment can only be said to affect the output if you squint really hard at the output - it might just be wishful thinking.
- the famous lines are continued in interesting, but clearly incoherent ways - _to be, or not to be: that is a man's sheep of more of property_ (k=25) - and sometimes, they even rhyme: _to be, or not to be: that is not be fear in me_ (k=10).
- the text actually seems better for lower k's - compare: [TODO]



## References

**Shakespeare Dataset**  
The Complete Works of William Shakespeare. (n.d.). Project Gutenberg. https://www.gutenberg.org/ebooks/100

**SMS Spam Collection Dataset**  
Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). SMS Spam Collection v.1. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
