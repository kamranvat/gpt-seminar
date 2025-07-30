# Report on the project 
## Contents:
### Task 1: 
- make shakespeare into test 1% and train
- train segmenter with varying k, try normalization strategies
- compare the performane against a different set
- figure out a measure for accuracy

### Task 2:
- Use cleaned shakespeare file / provided train-test-validation split
- use validation set to optimize hyperparameters in interpolation, not to optimize k
- develop n-gram engine based on BPE that can deal with different n (uni-, bi, tri, 4-gram), intrinsic evaluation for each:
    - report perplexity on BPE subwords 
        - for bigram: look at how different k's affect perplexity
    - laplace smoothing 
    - simple interpolation or backoff
- write program for extrinsic evaluation (gen. sentence from n-gram system)
    - give context
    - predict next word w. argmax or sampling
        - if word missing: avg. prob. of all or most likely word of unigram
        - use end-of-sequence tokens to determine generation stop

### Task 3 / softer version:
- implement early stopping w. patience
    - optional: optimize patience
    - save top k of model checkpoints
- tune hyperparams using grid search for each separately in this order: k, lr, interpolation weights
- try versions with different optimizers

## Introduction
Briefly introduce the project, its goals, and the overall structure of the report.

## Task 1: Data Preparation and Segmentation with BPE
### 1.1 Data Splitting
Describe the process of splitting the Shakespeare dataset into training and test sets.

### 1.2 Segmenter Training
Explain the training of the segmenter, including different values of k and normalization strategies.

### 1.3 Performance Comparison
Discuss how performance was compared across different settings and datasets.

### 1.4 Accuracy Measurement
Detail the chosen accuracy metric and its justification.

## Task 2: N-gram Engine with BPE
### 2.1 Data Usage and Splitting
Summarize the use of cleaned data and the provided splits.

### 2.2 Hyperparameter Optimization
Describe the use of the validation set for tuning interpolation hyperparameters.

### 2.3 N-gram Engine Implementation
#### 2.3.1 BPE Subword Modeling
Explain the BPE-based n-gram engine and its support for various n.

#### 2.3.2 Intrinsic Evaluation
Detail the intrinsic evaluation methods, including perplexity reporting and the effect of k.

#### 2.3.3 Smoothing and Interpolation
Describe the implementation of Laplace smoothing and interpolation/backoff strategies.

### 2.4 Extrinsic Evaluation
#### 2.4.1 Sentence Generation
Explain the program for generating sentences from the n-gram system.

#### 2.4.2 Prediction Strategies
Discuss context-based prediction, handling missing words, and stopping criteria.

## Task 3: Advanced Training Strategies
### 3.1 Early Stopping
Describe the implementation of early stopping and optional patience optimization.

### 3.2 Model Checkpointing
Explain how top k model checkpoints are saved.

### 3.3 Hyperparameter Tuning
Detail the grid search process for k, learning rate, and interpolation weights.

### 3.4 Optimizer Variants
Discuss experiments with different optimizers.

## Results
Summarize the key findings and results from each task.

## Discussion
Interpret the results, discuss challenges, and suggest possible improvements.

## Conclusion
Provide a brief conclusion and potential future work.

## References
List any references, datasets, or external resources used.
