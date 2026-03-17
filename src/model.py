import numpy as np

class Word2Vec:
    def __init__(
        self,
        epochs: int,
        vocab_size: int,
        embed_dim: int,
    ):
        # Hyperparameters
        self.epochs = epochs
        self.vocab_size = vocab_size  # The size of vocabulary
        self.embed_dim = embed_dim  # The dimension of the embedding
        
		# Initialize center_mat uniformly, and initialize context_mat to 0, per convention
        rng = np.random.default_rng(42)
        init_limit = 0.5 / embed_dim
        self.center_mat = rng.uniform(
            low=-init_limit, high=init_limit, size=(vocab_size, embed_dim)
        )
        self.context_mat = np.zeros_like((embed_dim, vocab_size))
