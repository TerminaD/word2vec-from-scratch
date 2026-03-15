Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.

Preferably, solutions should be provided as a link to a public GitHub repository.

1. What is word2vec? ✅
2. How is the training loop implemented in PyTorch? ✅ How are the files organized? ✅
3. Which word2vec variant? ✅ Which text dataset?
4. IN THE FAR FUTURE: optimizations! Accuracy/Performance
5. Flesh the thing out
   - Documentations
   - Environment file
6. Environment setup ✅

## Workflow
1. Preprocess dataset
	a. Vocab to word ID (integer) mapping
	b. Convert corpus to array of word IDs
	c. Put this into a script file

2. Forward training pass
	a. Parameters
		i. Center Word Matrix - W - V * d
		ii. Context Word Matrix - W' - d * V
	b. Training loop (with mini-batching)
		i. Center words: extract B * d from W (TODO: check if it's faster to extract rows or do matmul)
		ii. Positive context words: extract B * d from W'
		iii. Negative context words: B * k * d from W', sampled following *the unigram distribution of the training corpus raised to the 3/4 power* (TODO: check that)
		iv. Positive scores: dot product between "center words" and "positive context words" along the d-length axis, B * 1
		v. Negative scores: dot product between "center words" and "negative context words" along the d-length axis, B * k
	c. Objective: minimize the negative log-likelihood
	d. Hyper-parameters
		i. Batch size (B)
		ii. Epochs (E)
		iii. Vocab (cut-off, -1 for unlimited)
		iv. Embedding dimension (d)
		v. Negative-to-positive-context ratio (k)

3. Backward training pass
   	a. TODO: touch up on how this is done
   	b. Custom data structure for node in layer?

4. Inference
   	a. Print similar words to check correctness

5. Nice-to-haves
	a. Command-line arguments
	b. Check for pre-processing
	c. Progress bar & Tensorboard