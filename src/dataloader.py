import math
import numpy as np


class DataloaderSGNS:
    def __init__(
        self, 
        word_id_array: np.ndarray, 
        vocab_size: int,
        batch_size: int, 
        num_negative_samples: int,
        window_size: int,
        seed=42
      ):
        self.word_id_array = word_id_array

        if vocab_size == -1:
            raise Exception("dataloader.py: invalid vocab_size. Check word ID map file.")
        self.vocab_size = vocab_size
        self.corpus_size = np.size(word_id_array)
        print(f"dataloader.py: corpus_size set to {self.corpus_size}")
        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples  # Number of negative samples per (center, context) pair
        self.window_size = window_size      # The radius of the sliding window used to generate positive context words. If center word position is i and window_size is j, positions i-j, ..., i-1, i+1, ..., i+j would be context

        # Pre-calculate the 3/4th-power unigram distribution table for negative sampling
        unigram_dist = np.bincount(self.word_id_array) / len(self.word_id_array)
        raw_power_dist = np.power(unigram_dist, 0.75)
        self.neg_context_dist = raw_power_dist / np.sum(raw_power_dist)

        self.pos = 0	# Cursor for current position
        self.rng = np.random.default_rng(seed)


    def __len__(self):
        return math.ceil(self.corpus_size / self.batch_size)


    def __iter__(self):
        self.pos = 0
        self.order = self.rng.permutation(self.corpus_size)
        return self


    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pos >= self.corpus_size:
            raise StopIteration

        fin_pos = self.pos + self.batch_size
        curr_batch_size = self.batch_size
        if fin_pos > self.corpus_size:
            fin_pos = self.corpus_size
            curr_batch_size = fin_pos - self.pos

        center_idx = self.order[self.pos:fin_pos]
        context_offsets = np.concatenate([np.arange(-self.window_size, 0), np.arange(1, self.window_size + 1)])

        # Dynamic window radii
        sampled_radii = self.rng.integers(1, self.window_size + 1, size=curr_batch_size)

        I, J = np.meshgrid(center_idx, context_offsets)
        context_positions = I + J   # All context positions
        mask = (context_positions >= 0) & (context_positions < self.corpus_size) \
            & (np.abs(J) <= sampled_radii[np.newaxis, :])
        center_ids_array = self.word_id_array[I[mask]]
        pos_context_ids_array = self.word_id_array[context_positions[mask]]
        
        # TODO: For simplicity of implementation, center words and positive context words
        # are not excluded from negative words. This does not have a large impact for
        # large vocabulary sizes, but could be improved in the future.
        neg_context_ids_array = self.rng.choice(
            self.vocab_size,
            size=(np.size(center_ids_array), self.num_negative_samples),
            p=self.neg_context_dist,
        )

        self.pos = fin_pos
        return center_ids_array, pos_context_ids_array, neg_context_ids_array
    