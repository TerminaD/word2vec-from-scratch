import csv

input_path = "data/ws-353.csv"
output_path = "data/toy.txt"

words = []
with open(input_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for word1, word2, score in reader:
        if float(score) > 6:
            words.append(word1)
            words.append(word2)

with open(output_path, "w") as f:
    f.write(" ".join(words))