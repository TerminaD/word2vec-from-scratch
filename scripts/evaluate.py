import argparse
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import csv
from scipy import stats

from src.model import Word2VecSGNS

EVAL_SET_FULL_PATH = os.path.join("data", "ws-353.csv")

def build_parser():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Relative path to model checkpoint"
    )
    parser.add_argument(
        "--word_map_path",
        type=str,
        help="Relative path to word ID map file"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    data = np.load(args.model_path)
    vocab_size, embed_dim = data["center_mat"].shape
    model = Word2VecSGNS(vocab_size, embed_dim)
    model.center_mat = data["center_mat"]
    model.context_mat = data["context_mat"]
    
    with open(args.word_map_path, 'rb') as f:
        word_id_map = pickle.load(f)
    
    word_ids_1 = []
    word_ids_2 = []
    gt_scores = []
    skip_ctr = 0
    
    with open(EVAL_SET_FULL_PATH, newline='') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            word_1 = row[0]
            word_2 = row[1]
            gt_score = row[2]
            if word_1 not in word_id_map or word_2 not in word_id_map:
                skip_ctr += 1
                continue
            word_ids_1.append(word_id_map[word_1])
            word_ids_2.append(word_id_map[word_2])
            gt_scores.append(float(gt_score)) 
                   
    word_ids_1 = np.array(word_ids_1)
    word_ids_2 = np.array(word_ids_2)
    gt_scores = np.array(gt_scores)
    
    print(f"evaluate.py: vocabulary coverage is {word_ids_1.shape[0] / (word_ids_1.shape[0] + skip_ctr)}")
            
    embed_1 = model.get_embedding(word_ids_1)
    embed_2 = model.get_embedding(word_ids_2)
    denom = np.linalg.norm(embed_1, axis=1) * np.linalg.norm(embed_2, axis=1)
    cos_similar = np.sum(embed_1 * embed_2, axis=1) / (denom + 1e-8)
    res = stats.spearmanr(cos_similar, gt_scores).statistic

    print(f"evaluate.py: Spearman's correlation coefficient is {res}")

if __name__ == '__main__':
    main()