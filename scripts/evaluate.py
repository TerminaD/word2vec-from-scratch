import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pickle

from src.model import Word2VecSGNS
from src.evaluator import CosineSimilaritySpearmanEvaluator

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
        
    with open(EVAL_SET_FULL_PATH, newline='') as csv_f:
        evaluator = CosineSimilaritySpearmanEvaluator(csv_f, word_id_map)
        
    val = evaluator.evaluate(model)
    print(f"Spearman's: {val}")
    

if __name__ == '__main__':
    main()