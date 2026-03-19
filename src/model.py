import numpy as np

class Word2VecSGNS:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seed=42
    ):
        # Hyperparameters
        if vocab_size == -1:
            raise RuntimeError("model.py: vocab_size is -1. Potentially errors with reading word ID map file")
        self.vocab_size = vocab_size  # The size of vocabulary
        self.embed_dim = embed_dim  # The dimension of the embedding

        # Initialize weights uniformly
        rng = np.random.default_rng(seed)
        init_limit = 0.5 / embed_dim
        self.center_mat = rng.uniform(
            low=-init_limit, high=init_limit, size=(vocab_size, embed_dim)
        )
        self.context_mat = rng.uniform(
            low=-init_limit, high=init_limit, size=(vocab_size, embed_dim)
        )
        
        # Initialize internal states to None
        self._center_idx = None
        self._pos_idx = None
        self._neg_idx = None
        self._center_embeds = None
        self._pos_embeds = None
        self._neg_embeds = None
        self._pos_sigmoid = None
        self._neg_sigmoid = None
        self._center_grad = None
        self._pos_grad = None
        self._neg_grad = None

    def forward(self, center_idx, pos_idx, neg_idx) -> float:
        # Cached for backward pass
        self._center_idx = center_idx
        self._pos_idx = pos_idx
        self._neg_idx = neg_idx

        # Also cached
        self._center_embeds = self.center_mat[center_idx] # Actual batch size * embed_dim
        self._pos_embeds = self.context_mat[pos_idx]      # Actual batch size * embed_dim
        self._neg_embeds = self.context_mat[neg_idx]      # Actual batch size * num_negative_samples * embed_dim

        pos_scores = np.sum(self._center_embeds * self._pos_embeds, axis=1) # Actual batch size
        neg_scores = np.sum(self._center_embeds[:, np.newaxis, :] * self._neg_embeds, axis=2) # Actual batch size * num_negative_samples

        # Also cached
        self._pos_sigmoid = 1 / (1 + np.exp(np.clip(-pos_scores, -500, 500)))   # Actual batch size
        self._neg_sigmoid = 1 / (1 + np.exp(np.clip(neg_scores, -500, 500)))    # Actual batch size * num_negative_samples

        loss_array = -np.log(self._pos_sigmoid) - np.sum(np.log(self._neg_sigmoid), axis=1)
        loss_val = np.mean(loss_array)
        return loss_val


    def backward(self):
        if self._center_idx is None:
            raise RuntimeError("model.py: internal cache is not filled. Run forward() first. Aborting")

        pos_intermediate = self._pos_sigmoid - 1  # Actual batch size
        neg_intermediate = 1 - self._neg_sigmoid  # Actual batch size * num_negative_samples
        
        # Gradients are normalized by actual batch size, to match mean loss
        actual_batch_size = pos_intermediate.shape[0]

        self._center_grad = pos_intermediate[:, np.newaxis] * self._pos_embeds + \
            np.sum(neg_intermediate[:, :, np.newaxis] * self._neg_embeds, axis=1)  # Actual batch size * embed_dim
        self._center_grad /= actual_batch_size
            
        # Actual batch size * embed_dim
        self._pos_grad = pos_intermediate[:, np.newaxis] * self._center_embeds / actual_batch_size
        
        # Actual batch size * num_negative_samples * embed_dim
        self._neg_grad = neg_intermediate[:, :, np.newaxis] * self._center_embeds[:, np.newaxis, :] / actual_batch_size


    def get_embedding(self, word_idx):
        return self.center_mat[word_idx]


    def get_parameters(self):            
        return {
            'center_mat': self.center_mat,
            'context_mat': self.context_mat,
        }
