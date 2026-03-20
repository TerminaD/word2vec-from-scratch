import argparse
import numpy as np
import os
import pickle

DATASET_PATH = "data/text8.txt"

def build_parser():
	parser = argparse.ArgumentParser(description="Dataset preprocessing script")
	parser.add_argument("--corpus_size", type=int, default=0,
                     help="The number of words at the start of the corpus that are preprocessed. Default to 0. If set to 0 or negative, the entire corpus is preprocessed.")
	parser.add_argument("--min_frequency", type=int, default=0, 
                     help="The minimum number of times a word need to appear to be included in the vocabulary. If set to 0 or negative, all words are included.")
	return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    preproc_result_path = f"data/cs{args.corpus_size}-mf{args.min_frequency}"
    if os.path.exists(preproc_result_path):
        print("Preprocessing has already been done with the current configuration. Exiting")
        return
    os.mkdir(preproc_result_path)
    
    word_list = []
    with open(DATASET_PATH) as f:
        word_list = f.read().split()
    
    if args.corpus_size > 0:
        word_list = word_list[:args.corpus_size]
    
    word_id_map = {}
    curr_word_id = 0
    
    if args.min_frequency > 1:
        freq_map = {}
        for word in word_list:
            if word in freq_map:
                freq_map[word] += 1
            else:
                freq_map[word] = 1

        for word, freq in freq_map.items():
            if freq >= args.min_frequency:
                word_id_map[word] = curr_word_id
                curr_word_id += 1

        print(f"Ratio of vocabulary kept: {curr_word_id / len(freq_map)}")

    else:
        curr_word_id = 0
        for word in word_list:
            if word in word_id_map:
                continue
            word_id_map[word] = curr_word_id
            curr_word_id += 1

    word_id_list = [word_id_map[word] for word in word_list if word in word_id_map]
    word_id_array = np.array(word_id_list)

    with open(f"{preproc_result_path}/word_id_map.pkl", "wb") as map_file:
        pickle.dump(word_id_map, map_file)

    with open(f"{preproc_result_path}/word_id_array.npy", "wb") as array_file:
        np.save(array_file, word_id_array)
    

if __name__ == "__main__":
    main()
    