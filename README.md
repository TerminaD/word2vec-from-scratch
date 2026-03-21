This repo contains training and evaluation code for a skip-gram word2vec model with negative sampling (SGNS word2vec). No machine learning packages are used in this project, just NumPy.

# How to Use
## 1. Set up Environment
The package management for this project is done with `conda`. To install the requisite packages, run:
```bash
conda env create -f environment.yml
```

And activate the environment with:
```bash
conda activate jetbrains-word2vec
```

Note that although PyTorch is a dependency, no actual machine learning features are used. It is only used for writing statistics to TensorBoard.

## 2. Download The text8 Dataset
Due to the large size of the text8 dataset (around 100MB), it is not included in the GitHub repository. Instead, you can run the following command to download it:
```bash
python scripts/download_text8.py
```

## 3. Preprocess The Dataset
Before being used for training, the dataset is preprocessed using:
```bash
python scripts/preprocess.py --corpus_size=0 --min_frequency=5 --subsample_threshold=1e-5
```

For more information about the arguments, run:
```bash
python scripts/preprocess.py -h
```

## 4. Train The Model
```bash
python scripts/train.py --preproc_dir_name=cs0-mf5-st1e-05 --epoch=5 --batch_size=32 --initial_lr=2 --final_lr=0.001 --num_neg_samples=5 --window_size=5 --embed_dim=100
```

For more information about the arguments, run:
```bash
python scripts/train.py -h
```

After or during training, you can monitor the progressing of training loss and Spearman's correlation coefficient (SCC, my evaluation metric of choice) with TensorBoard. To start the TensorBoard web app, run:
```bash
tensorboard --logdir=runs/
```

After that, go to `http://localhost:6006/` to access the web app.

## 5. Evaluate The Model
A script is provided for calculating SCC on a pre-trained model file. Run:
```bash
python scripts/evaluate.py --model_path=RELATIVE/PATH/TO/MODEL/FILE --word_map_path=RELATIVE/PATH/TO/WORD/MAP/FILE
```

A demo is available with:
```bash
python scripts/evaluate.py --model_path=models/good/model.npz --word_map_path=models/good/word_id_map.pkl
```

# Project Structure
For clarity and ease of understanding, this project adopts a PyTorch-like file structure, in which the model, dataloader and optimizer are separate classes.

```
word2vec/
├── .gitignore
├── README.md
├── environment.yml
├── data/						# Datasets and preprocessing results
│   ├── text8.txt					# Training set: text8
│   ├── ws-353.csv					# Evaluation set: WordSimilarity-353
│   └── cs0-mf5-st1e-05/			# Demo preprocessing results
│       ├── word_id_array.npy
│       └── word_id_map.pkl
├── models/						# Pre-trained model checkpoints
│   └── demo/						# Demo
│       ├── model.npz					
│       └── word_id_map.pkl
├── runs/						# TensorBoard metric logs
│   └── demo/						# Demo
│       └── events.out.tfevents
├── scripts/					# Scripts for a certain action
│   ├── download_text8.py			# Downloads the text8 dataset
│   ├── evaluate.py					# Evaluates a pre-trained model file
│   ├── preprocess.py				# Preprocesses the text8 dataset
│   └── train.py					# Trains the word2vec model
└── src/						# Provides classes to support scripts
    ├── __init__.py
    ├── dataloader.py				# Dataloader, generates batched word ID pairs
    ├── evaluator.py				# Evaluator
    ├── model.py					# Model structure, forward and backward behavior
    └── optimizer.py				# Optimizer, used to update model weights
```

# Method
## Skip-Gram word2vec with Negative Sampling: An Overview

## Dataset and Metric Selection
### text8
The training set used for this project is text8, which is a partial text dump of Wikipedia. It has a word count of around 17 million. It is cleaned into lower-case words, delimited by whitespaces.

text8 is chosen for: 1. Its size strikes a nice balance between the amount of semantic information and training time. 2. Its format allows for simple preprocessing, eliminating the need for cleaning. 

The dataset file used for this project is downloaded at http://mattmahoney.net/dc/text8.zip on 15/03/2026 10:58 PM. However, as the text8 dataset is no longer a work in progress, the dataset file should not change in the foreseeable future.

### WordSimilarity-353
As the skip-gram word2vec model itself does not have an evaluation metric, a downstream task, consisting of an evaluation set and an evaluation metric is needed to evaluate model performance. TODO

## Dataset Preprocessing

## Model Architecture

## Forward Pass and Loss Function

## Backward Pass

## Parameter Updates

## Evaluation

