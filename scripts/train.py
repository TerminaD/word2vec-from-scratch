import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

from src.dataloader import Dataloader

DATA_DIR = "data"
ARRAY_FILE_NAME = "word_id_array.npy"
MAP_FILE_NAME = "word_id_map.pkl"

def build_parser():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--preproc_dir_name",
        type=str,
        help="The directory name containing the preprocessing artifacts. Do not include the data directory name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of distinct center words per training batch. Note that this is not the length of the 0-th axis of input arrays.",
    )
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        help="The number of (center word, negative context word) pairs generated per (center word, positive context word) pair.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="The radius of the positive context window. If center word position is i and window_size is j, positions i-j, ..., i-1, i+1, ..., i+j would be the context.",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    array_path = os.path.join(DATA_DIR, args.preproc_dir_name, ARRAY_FILE_NAME)
    if not os.path.exists(array_path):
        print("Preprocessing not done. Aborting")
        return
    map_path = os.path.join(DATA_DIR, args.preproc_dir_name, MAP_FILE_NAME)
    word_id_array = np.load(array_path)
    vocab_size = -1
    with open(map_path, 'rb') as map_f:
        word_id_map = pickle.load(map_f)
        vocab_size = max([word_id for _, word_id in word_id_map]) + 1
    dataloader = Dataloader(word_id_array, vocab_size, args.batch_size, args.num_neg_samples, args.window_size)
    
    for center_idx, pos_idx, neg_idx in tqdm(dataloader, desc="Per-batch progress"):
        

if __name__ == "__main__":
    main()
