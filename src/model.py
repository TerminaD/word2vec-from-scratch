import numpy as np

class Word2VecSkipGramNegativeSampling:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
    ):
        # Hyperparameters
        self.vocab_size = vocab_size  # The size of vocabulary
        self.embed_dim = embed_dim  # The dimension of the embedding
        
		# Initialize center_mat uniformly, and initialize context_mat to 0
        rng = np.random.default_rng(42)
        init_limit = 0.5 / embed_dim
        self.center_mat = rng.uniform(
            low=-init_limit, high=init_limit, size=(vocab_size, embed_dim)
        )
        self.context_mat = rng.uniform(
            low=-init_limit, high=init_limit, size=(vocab_size, embed_dim)
        )

    def forward(self, center_idx, pos_idx, neg_idx) -> float:
        pass
    
    def backward(self, center_idx, pos_idx, neg_idx):
        pass
    
    def get_embedding(self, word_idx):
        pass
    
    def get_parameters(self):
        pass