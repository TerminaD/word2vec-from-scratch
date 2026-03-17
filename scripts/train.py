DATA_DIR = "data"
ARRAY_FILE_NAME = "word_id_array.npy"
MAP_FILE_NAME = "word_id_map.pkl"
        
array_path = os.path.join(DATA_DIR, preproc_dir_name, ARRAY_FILE_NAME)
map_path = os.path.join(DATA_DIR, preproc_dir_name, MAP_FILE_NAME)
        
word_id_array = np.load(array_path)
word_id_map = None
with open(map_path, 'rb') as map_f:
    word_id_map = pickle.load(map_f)