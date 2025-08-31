REPORT
Our implementation demonstrates the evolution of natural language processing techniques, from simple statistical methods to more sophisticated neural approaches. The code is structured modularly, with separate implementations for BPE tokenization, classical n-gram models, and neural n-gram networks.


Task 0: Text Analysis
To get an overview of the data, we implemented both Unix-style commands and Python equivalents for analyzing the initial Shakespeare corpus. The analysis revealed the fundamental characteristics of the text that would inform our subsequent modeling decisions.
0.1 Space-based Tokenization
The initial corpus analysis was performed using space-based tokenization to understand word frequency distributions. We implemented both shell-style and Python approaches:
Unix command equivalent:
bashtr -sc 'A-Za-z' '\n' < shakespeare.txt | sort | uniq -c |sort -n -r
Python implementation:

pythonimport re
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
The analysis revealed the most frequent tokens in the original corpus:
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
This frequency distribution follows the expected Zipfian distribution typical of natural language, with function words dominating the vocabulary.


Task 1: Data Preparation and Segmentation with BPE

1.1 Data Splitting and Cleaning
The initial corpus required substantial preprocessing to remove metadata and licensing information that appeared throughout the text. Our preprocessing pipeline included:

Whitespace normalization: Collapsing multiple whitespace characters into single spaces using ' '.join(corpus.split())
Case normalization: Converting to lowercase using corpus.casefold() for better Unicode handling
Corpus splitting: Extracting test sets as random samples of specified percentages

The cleaning process significantly reduced corpus size:

Original corpus: 4,941,133 characters (901,325 words)
Cleaned corpus: 1,041,007 characters (190,999 words)

A cleaned version of the corpus with proper train/test/validation splits was provided by course participant Mohamed Ebrahim, which we used for subsequent experiments.


1.2 BPE Implementation
We implemented Byte Pair Encoding (BPE) as a subword tokenization method. The algorithm follows these key steps:

Initialization: Start with individual characters as the initial vocabulary
Pair counting: Count all adjacent token pairs in the corpus
Merging: Replace the most frequent pair with a new merged token
Iteration: Repeat until k merges are completed

Our implementation supports both serial and parallelized processing:
pythondef get_most_frequent_pair(self, corpus):
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


1.3 Performance Evaluation
We evaluated BPE performance across different test sets to measure generalization:

In-domain (ID) test: Shakespeare test set from the same source
Out-of-domain (OOD) test: SMS dataset with different linguistic characteristics
Coverage analysis: Percentage of text covered by tokens of various lengths

The evaluation used coverage metrics, measuring how well the learned vocabulary segments unseen text. We tested multiple values of k (number of merges) ranging from 100 to 10,000.


1.4 Accuracy Measurement
We measured segmentation quality using coverage percentages - the fraction of characters in test text that could be represented by tokens of minimum length n. This metric captures how well the BPE vocabulary generalizes to new text while avoiding over-segmentation.
The BPE class includes comprehensive evaluation methods:
pythondef evaluate(self, vocab, test_set, max_n=3):
    # check percentage of text covered by all tokens, then with increasing n
    coverages = []
    matched_chars = []
    
    for n in range(1, max_n + 1):
        t, coverage, m = self.test(vocab, test_set, min_token_length=n)
        coverages.append(coverage)
        matched_chars.append(m)
    
    return coverages


Task 2: N-gram Engine
2.1 Data Usage and Validation Strategy
We utilized the cleaned Shakespeare corpus with proper train/test/validation splits for all n-gram experiments. The validation set served specifically for hyperparameter optimization in interpolation models, while k values were treated as a separate experimental variable.

2.2 N-gram Engine Implementation
Our n-gram implementation supports arbitrary order n and includes sophisticated smoothing techniques. The core architecture uses matrices to store frequencies and probabilities:

pythonclass NGram:
    def _init_(self, vocab, n=4, laplace_smoothing=True, interpolation=False, backoff=False, lambdas=None):
        self.n = n
        self.n_gram_contexts = []  # contexts for each n-gram level
        self.vocab = vocab
        self.n_gram_probabilities = []  # probability matrices
        self.n_gram_frequencies = []   # frequency matrices


2.2.1 BPE Integration
The n-gram engine operates on BPE-tokenized text, allowing for subword-level language modeling. This approach combines the benefits of subword regularization with statistical language modeling:
pythondef get_n_gram_counts(self, train, n):
    ngram_counts = Counter(
        tuple(train[i:i + n])
        for i in range(len(train) - n + 1)
    )
    return ngram_counts


2.2.2 Smoothing Techniques
We implemented multiple smoothing strategies:
Laplace Smoothing: Adds a constant to all frequency counts to handle zero probabilities

pythonif self.laplace_smoothing:
    self.n_gram_frequencies[n] += 1
Linear Interpolation: Combines probabilities from different n-gram orders
pythondef get_interpolated_probabilities(self, test, lambdas):
    # Combines unigram through n-gram probabilities using learned weights
    for n in range(self.n):
        p += lambdas[n] * self.n_gram_probabilities[n][context, token]
Backoff: Falls back to lower-order models when higher-order contexts are unseen


2.2.3 Hyperparameter Optimization
Lambda weights for interpolation are optimized using random search on the validation set:
pythondef optimize_lambdas(self, validation, strategy="random", samples=1000):
    # Random search for optimal interpolation weights
    best_lambdas = [-1, -1, -1]
    best_probability = -np.inf
    
    for _ in range(samples):
        lambdas = rng.random(self.n)
        lambdas = lambdas/np.sum(lambdas)  # normalize
        probability = self.get_interpolated_probabilities(validation, lambdas)
        
        if probability > best_probability:
            best_lambdas = lambdas
            best_probability = probability


2.3 Intrinsic Evaluation
2.3.1 Perplexity Computation

Perplexity serves as our primary intrinsic evaluation metric:
pythondef test_perplexity(self, test):
    probability = self.get_probability(test)
    return np.power(2, -probability/len(test))
We systematically evaluated how different k values (BPE merge operations) affect perplexity across n-gram orders, with particular focus on bigrams due to their balance of context and computational efficiency.


2.3.2 Multi-order Analysis
We tested n-gram orders from 1 to 4, examining how increased context length affects modeling performance. Higher-order models capture more linguistic structure but suffer from increased sparsity.


2.4 Extrinsic Evaluation
2.4.1 Text Generation
Our n-gram models support both greedy and sampling-based text generation:
pythondef predict(self, context, max_length=10, method="greedy", end_of_sequence_tokens=['.', '!', '?']):
    # Generate text using either greedy selection or probabilistic sampling
    while next_token not in end_of_sequence_tokens and len(sequence) < max_length:
        if method == "greedy":
            best_index = np.argmax(probabilities)
        elif method == "sample":
            best_index = np.random.choice(np.arange(len(self.vocab)), p=probabilities)
        
        next_token = self.vocab[best_index]
        sequence.append(next_token)


2.4.2 Context Handling
The generation system handles variable-length contexts and implements fallback strategies for unseen n-grams. End-of-sequence tokens (periods, exclamation marks, question marks) provide natural stopping criteria for generated text.


Task 3: Neural N-Gram
3.1 Neural Architecture
We implemented a neural n-gram model using PyTorch, representing contexts as learned embeddings:
pythonclass NeuroNgram(nn.Module):
    def _init_(self, vocab, n=2):
        super()._init_()
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size ** (self.n - 1), self.vocab_size)
The model encodes n-gram contexts as single indices by treating them as base-vocab_size numbers, enabling direct embedding lookup.


3.2 Training Infrastructure
3.2.1 Early Stopping Implementation
We implemented early stopping with configurable patience to prevent overfitting:

pythondef train(model, data, writer, optimizer=None, patience=5, validate_every_x=1):
    steps_without_validation_improvement = 0
    best_valid_loss = torch.inf
    
    for step in range(steps):
        # Training step
        if validation_data is not None and step % validate_every_x == 0:
            _, loss, pp = model(x, y)
            if loss < best_valid_loss:
                best_valid_loss = loss
                steps_without_validation_improvement = 0
                # Save model checkpoint
            else:
                steps_without_validation_improvement += 1
            
            if steps_without_validation_improvement >= patience:
                logger.info(f"Early stopping triggered at step {step}")
                break


3.2.2 Model Checkpointing
The system saves top-k model checkpoints based on validation performance, automatically managing storage by removing older checkpoints when the limit is exceeded.


3.3 Hyperparameter Optimization
We conducted systematic hyperparameter exploration across multiple dimensions:

Optimizer Variants: We tested 12 different optimizer configurations:

SGD with and without momentum
Adam with learning rates: 1e-4, 1e-2, 1e-1, 25e-2, 5e-1
AdamW with learning rates: 1e-4, 1e-2, 1e-1, 25e-2, 5e-1

BPE Vocabulary Size (k): Tested values from 100 to 10,000 merges to study the effect of subword vocabulary size on neural model performance.
Architecture Parameters: Fixed n=2 for systematic comparison, with configurable context and batch sizes.


3.4 Evaluation and Generation
The neural n-gram model supports the same evaluation metrics as classical n-grams:
pythondef evaluate_perplexity_on_test(self, tokenized_test, batch_size=None, context_size=None):
    # Compute perplexity on test data using the trained neural model
    logits, loss, pp = self.forward(context=x, target=y)
    return pp
Text generation uses probabilistic sampling from the learned distribution:
pythondef predict(self, context, max_new_tokens=50):
    for i in range(max_new_tokens):
        logits, loss, _ = self(encoded_context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_context = torch.multinomial(probs, num_samples=1)
        prediction = torch.cat((prediction, next_context), dim=1)
    return prediction


Experimental Results
Our experiments revealed several key insights:

BPE Vocabulary Size: Larger vocabularies (higher k) generally improved performance up to a point, with diminishing returns beyond k=2000-5000 depending on the model type.
N-gram Order: Classical n-grams showed better performance around n=3-4, balancing context richness with sparsity issues.

Smoothing Effectiveness: Linear interpolation consistently outperformed simple Laplace smoothing, with optimized lambda weights providing substantial improvements.

Neural vs. Classical: Neural n-gram models showed promise but required careful hyperparameter tuning to match classical model performance.

Generalization: All models showed expected degradation when tested on out-of-domain data (SMS vs. Shakespeare), with BPE helping to maintain some robustness.

## Milestone 4: GPT

Our journey through natural language processing is nearing it's end and brought us to implement a full GPT-style-transformer.

Building the Neural Architecture

Our GPT implementation centers around the self-attention mechanism, implemented through individual attention heads:

class Head(nn.Module):
    """One head of masked self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

We think of it as a more sophisticated conversation system. Each token asks questions (queries) about what it's looking for, while all previous tokens offer answers (keys) about what information they contain. The actual content that gets passed forward (values) is then weighted by how well the questions match the answers.

The triangular mask ensures our model plays by the rules - it can't peek into the future during training, maintaining the sequential nature so that the language modeling becomes meaningful.

So why stop at one conversation? When our multi-head attention can run in parallel:

class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)

After attention determines what's relevant, feed-forward networks process this information:

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

The expansion to four times the embedding dimension provides computational space for complexer transformations, while the contraction back maintains dimensional consistency throughout the network.
So to bring it all together 
Each transformer block elegantly combines these components: 

class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Self-attention with residual connection
        x = x + self.ff(self.ln2(x))  # Feed-forward with residual connection
        return x

The residual connections (x = x + ...) should solve the vanishing gradient problem, allowing us to stack many layers while maintaining stable learning. These connections provide "gradient highways" that enable information and learning signals to flow efficiently through deep networks.
Our complete GPT model orchestrates all these components into a cohesive system:

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

The architecture is beautifully logical (are we bragging? maybe. is it deserved? i don't know): token embeddings provide semantic meaning, position embeddings add sequential awareness, transformer blocks perform the heavy lifting of pattern recognition and context integration, and the final layer translates back to vocabulary predictions.

Then lets move to our Seamless Integration with our BPE foundation.
One of the satisfying aspects of our implementation is how seamlessly it integrates with the BPE tokenization system we built earlier. Our data preparation pipeline transforms Shakespeare's prose into the numerical sequences our transformer craves:

(code)
 encoder_gpt.py - The bridge between BPE and neural networks
bpe = BPE()
bpe.set_vocab(FileUtils().load_vocab(vocab_path))
seg_tokens_train = FileUtils().load_vocab(segmented_path_train)
train_ids = bpe.encode(seg_tokens_train)

 Efficient storage for training
dtype = np.uint16 if vocab_size < (1 << 16) else np.uint32
np.array(train_ids, dtype=dtype).tofile(str(data_dir / "train.bin"))

This integration should ensure that our transformer operates on the same meaningful subword units that proved effective in our earlier n-gram experiments. The vocabulary size optimization demonstrates attention to computational efficiency - using 16-bit integers when possible to reduce memory usage and improve training speed.

Training the Neural Bard
Training a transformer requires infrastructure to handle the computational demands. Our batch processing system efficiently samples training sequences:

def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor, device: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

This sliding window approach ensures our model sees diverse examples while maintaining the sequential relationships essential for language learning.

Small Model (64 dimensions, 4 heads, 4 layers): Perfect for experimentation and understanding the architecture. Fast training allows rapid iteration and debugging.
Medium Model (256 dimensions, 4 heads, 6 layers): Our recommended configuration, striking an optimal balance between performance and computational efficiency. This became our primary experimental setup.
Large Model (256 dimensions, 16 heads, 12 layers): Pushing toward maximum performance within our computational constraints, demonstrating how the architecture scales.
The training loop incorporates modern best practices:

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

HOW TO GENARATE
once trained our GPT transformes ... (pun intended) into a digital Shakespeare capable of generating new nonsense through sophisticated sampling strategies. The genaration system offers control over creativity and coherence:

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

        Temperature controls the model's creativity - low values produce conservative, predictable continuations, while higher values encourage let's call them bold and surprising choices. Top-k sampling focuses attention on the most promising options, while nucleus (top-p) sampling dynamically adjusts the vocabulary size based on probability mass, providing more nuanced control over generation diversity.
        
        The End :)
        This was the End of the Report from group 20
## References

**Shakespeare Dataset**  
The Complete Works of William Shakespeare. (n.d.). Project Gutenberg. https://www.gutenberg.org/ebooks/100

**SMS Spam Collection Dataset**  
Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). SMS Spam Collection v.1. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
