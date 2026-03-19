import numpy as np

class SGDDecayOptimizer:
    def __init__(self, model, initial_lr, final_lr):
        self.model = model
        self.initial_lr = initial_lr
        self.final_lr = final_lr

    def step(self, progress):
        m = self.model
        lr = self.initial_lr * (1 - progress) + self.final_lr * progress

        # np.add.at is used instead of square bracket indexing, as it allows duplicate indices
        np.add.at(m.center_mat, m._center_idx, -lr * m._center_grad)
        
        np.add.at(m.context_mat, m._pos_idx, -lr * m._pos_grad)
        
        neg_idx_flat = m._neg_idx.ravel()
        neg_grad_flat = m._neg_grad.reshape(-1, m.embed_dim)
        np.add.at(m.context_mat, neg_idx_flat, -lr * neg_grad_flat)
