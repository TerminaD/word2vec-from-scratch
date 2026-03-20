import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil

from src.dataloader import DataloaderSGNS
from src.model import Word2VecSGNS
from src.optimizer import SGDDecayOptimizer

DATA_DIR = "data"
MODELS_DIR = "models"
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
        "--epoch",
        type=int,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of distinct center words per training batch. Note that this is not the length of the 0-th axis of input arrays.",
    )
    parser.add_argument(
        "--initial_lr",
        type=float,
        help="Initial learning rate for the optimizer, which uses linear LR decay"
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        help="Final learning rate for the optimizer, which uses linear LR decay"
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
    parser.add_argument(
        "--embed_dim",
        type=int,
        help="Dimension of the embedding vector",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    array_path = os.path.join(DATA_DIR, args.preproc_dir_name, ARRAY_FILE_NAME)
    if not os.path.exists(array_path):
        raise RuntimeError("Preprocessing not done before training. Aborting")
    map_path = os.path.join(DATA_DIR, args.preproc_dir_name, MAP_FILE_NAME)
    
    word_id_array = np.load(array_path)
    vocab_size = -1
    word_id_map = None
    with open(map_path, 'rb') as map_f:
        word_id_map = pickle.load(map_f)
        vocab_size = max([word_id for _, word_id in word_id_map.items()]) + 1
    
    model = Word2VecSGNS(vocab_size, args.embed_dim)
    optimizer = SGDDecayOptimizer(model, args.initial_lr, args.final_lr)
    dataloader = DataloaderSGNS(word_id_array, vocab_size, args.batch_size, args.num_neg_samples, args.window_size)
    
    writer = SummaryWriter(f"runs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    
    total_batches = len(dataloader) * args.epoch
    global_batch_idx = 0

    for e in tqdm(range(args.epoch), desc="Epochs"):
        loss_list = []
        for center_idx, pos_idx, neg_idx in tqdm(dataloader, desc="Batches"):
            loss_list.append(model.forward(center_idx, pos_idx, neg_idx))
            model.backward()
            progress = global_batch_idx / total_batches
            optimizer.step(progress)
            global_batch_idx += 1
                
        avg_loss = sum(loss_list) / len(loss_list)
        writer.add_scalar("Loss/train", avg_loss, e)
    
    
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    curr_models_dir = os.path.join(MODELS_DIR, time_stamp)
    os.makedirs(curr_models_dir, exist_ok=True)
    save_path = os.path.join(curr_models_dir, "model.npz")
    np.savez(save_path, **model.get_parameters())
    shutil.copy(map_path, os.path.join(curr_models_dir, MAP_FILE_NAME))
    print(f"Model and word map saved")


if __name__ == "__main__":
    main()
