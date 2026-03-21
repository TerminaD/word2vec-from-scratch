import numpy as np
import csv
from scipy import stats


class CosineSimilaritySpearmanEvaluator:
    def __init__(self, csv_file, word_id_map):
        word_ids_1 = []
        word_ids_2 = []
        gt_scores = []
        skip_ctr = 0
        
        reader = csv.reader(csv_file)
        _ = next(reader)
        for row in reader:
            word_1 = row[0]
            word_2 = row[1]
            gt_score = float(row[2])
            if word_1 not in word_id_map or word_2 not in word_id_map:
                skip_ctr += 1
                continue
            word_ids_1.append(word_id_map[word_1])
            word_ids_2.append(word_id_map[word_2])
            gt_scores.append(gt_score) 
                   
        self.word_ids_1 = np.array(word_ids_1)
        self.word_ids_2 = np.array(word_ids_2)
        self.gt_scores = np.array(gt_scores)
        print(f"evaluator.py: vocabulary coverage is {self.word_ids_1.shape[0] / (self.word_ids_1.shape[0] + skip_ctr)}")
        
        
    def evaluate(self, model):
        embed_1 = model.get_embedding(self.word_ids_1)
        embed_2 = model.get_embedding(self.word_ids_2)
        denom = np.linalg.norm(embed_1, axis=1) * np.linalg.norm(embed_2, axis=1)
        cos_similar = np.sum(embed_1 * embed_2, axis=1) / (denom + 1e-8)
        return stats.spearmanr(cos_similar, self.gt_scores).statistic
