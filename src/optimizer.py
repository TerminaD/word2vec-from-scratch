import numpy as np

class SGDOptimizer:
    def __init__(self, model, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.model = model
        self.momentum = momentum
        
        # Velocity matrices
        self.center_vel = np.zeros_like(model.center_mat)    
        self.context_vel = np.zeros_like(model.context_mat)

    def step(self):
        m = self.model
        lr = self.learning_rate
        mu = self.momentum

        # Build dense gradient arrays to make velocity updates easier
        center_grad_dense = np.zeros_like(m.center_mat)
        context_grad_dense = np.zeros_like(m.context_mat)

        # Use np.add.at instead of square bracket indexing to handle duplicate indices
        np.add.at(center_grad_dense, m._center_idx, m._center_grad)
        np.add.at(context_grad_dense, m._pos_idx, m._pos_grad)
        neg_idx_flat  = m._neg_idx.ravel()
        neg_grad_flat = m._neg_grad.reshape(-1, m.embed_dim)
        np.add.at(context_grad_dense, neg_idx_flat, neg_grad_flat)

        self.center_vel = mu * self.center_vel + center_grad_dense
        self.context_vel = mu * self.context_vel + context_grad_dense

        m.center_mat -= lr * self.center_vel
        m.context_mat -= lr * self.context_vel
        