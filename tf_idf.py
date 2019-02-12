import os
import numpy as np
import operator
import random
import json
# import nltk

NUM_WORDS = 300
SHUFFLE_SEED = 4


def preprocess_line(line):
    return list(line.lower().split(' '))


positive_instances = list()
for file in os.listdir(os.path.join('train', 'pos')):
    with open(os.path.join('train', 'pos', file)) as f:
        positive_instances.append(f.read())

negative_instances = list()
for file in os.listdir(os.path.join('train', 'neg')):
    with open(os.path.join('train', 'neg', file)) as f:
        negative_instances.append(f.read())

total_instances = positive_instances + negative_instances

num_docs = len(total_instances)
word_counts = dict()
doc_occurences = dict()
# for now just creating words by splitting on whitespace
for line in total_instances:
    tokens = preprocess_line(line)
    curr_tokens = set()
    for t in tokens:
        if t in word_counts.keys():
            word_counts[t] += 1
            if t not in curr_tokens:
                doc_occurences[t] += 1
                curr_tokens.add(t)  # we do not count a document twice for words appearing multiple times in a sample
        else:
            word_counts[t] = 1
            doc_occurences[t] = 1


# sorting words by frequency, taking the top N (for now 300)
top_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)[:NUM_WORDS]

# mix up the positive and negative training examples
random.seed(SHUFFLE_SEED)
random.shuffle(total_instances)

# creating a list of processed training examples, with each word converted to its tf-idf score
labelled_samples = list()
for sample in total_instances:
    word_vector = [0.0] * NUM_WORDS
    tokens = preprocess_line(sample)
    for token in tokens:
        word_index = 0
        for tup in top_counts:
            if token == tup[0]:
                tf_idf = tokens.count(token) * np.log(num_docs / doc_occurences[token])
                word_vector[word_index] = tf_idf
                break
            else:
               word_index += 1

    # new_sample['tf_idf_word_vector'] = word_vector
    label = 0.0
    if sample in positive_instances:
        label = 1.0

    new_sample = np.array([np.array(word_vector), np.array(label)])
    labelled_samples.append(new_sample)
print(labelled_samples[0])

with open('train_tfidf.json', 'w') as fp:
    json.dump(labelled_samples, fp)


