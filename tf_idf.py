import os
import numpy as np
import operator
import random
import json
import csv
import re
import nltk
from tqdm import trange

NUM_WORDS = 300
SHUFFLE_SEED = 4

def has_some_alphanumeric_characters( line ):
    """
    Re wrapper returning True/False depending on whether the input line (string) contains at least one alphabetic character.
    
    Input:
        
        line: a string.
        
    Returns:
    
        boolean True/False
        
    """
    if re.search('[a-zA-Z]', line):
        return True
    else:
        return False

def word_tokenize( line: str ):
    """
    Tokenizes input string into words (where each word has >= 1 letter).
    
    Input:
        
        line: string to tokenize into words.
            
    Returns: 
        
        a list of all tokens with >= 1 alphabetic character.
        
    """
    return [ word for word in nltk.word_tokenize( line ) if has_some_alphanumeric_characters( word ) ]


def preprocess_line(line):
    return list(line.lower().split(' '))


positive_instances = list()
for file in os.listdir(os.path.join('train', 'pos')):
    with open(os.path.join('train', 'pos', file), encoding="utf8") as f:
        positive_instances.append(f.read())

negative_instances = list()
for file in os.listdir(os.path.join('train', 'neg')):
    with open(os.path.join('train', 'neg', file), encoding="utf8") as f:
        negative_instances.append(f.read())

total_instances = positive_instances + negative_instances

num_docs = len(total_instances)
word_counts = dict()
doc_occurences = dict()

# for now just creating words by splitting on whitespace
print( "\nPreprocessing lines" )
for line_index in trange( len( total_instances ) ):
    line = total_instances[ line_index ]
#for line in total_instances:
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
print( "\nVectorizing" )
for sample_index in trange( len( total_instances ) ):
    sample = total_instances[sample_index]
#for sample in total_instances:
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

    # new_sample = np.array([np.array(word_vector), np.array(label)])
    new_sample = [word_vector, label]
    labelled_samples.append(new_sample)
# print(labelled_samples[0])

with open('train_tfidf.txt', 'w', encoding="utf8") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(labelled_samples)

