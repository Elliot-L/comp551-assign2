import os, operator, random, json, csv, re, nltk, pickle

import numpy as np

from scipy.sparse import csr_matrix
from nltk.stem import WordNetLemmatizer
from tqdm import trange
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

NUM_WORDS = 252192  # 300
SHUFFLE_SEED = 4

# Analyser function for the count vectorizer to store ? and !

def words_and_char_bigrams(text):
    words = re.findall(r'\w{1,}|!|\?|[.]{2,}', text)
    for w in words:
        yield w

def has_some_alphanumeric_characters(line):
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


def word_tokenize(line: str):
    """
    Tokenizes input string into words (where each word has >= 1 letter).

    Input:

        line: string to tokenize into words.

    Returns:

        a list of all tokens with >= 1 alphabetic character.

    """
    return [word for word in nltk.word_tokenize(line) if has_some_alphanumeric_characters(word)]


def decontract(line: str):
    """
    Re wrapper to expand contractions (e.g. 'would_n't_' -> 'would not').
    Arguments:
        line: string to decontract.
    Returns:
        input string without contractions.
    """
    re.sub(r"n't", r" not ", line)


def lemmatize(word: str, lemmatizer=None):
    """
    nltk.WordNetLemmatizer() wrapper to lemmatize a word.
    Arguments:
        word: string version of word to lemmatize.
        lemmatizer: None or some nltk.stem.wordnet.WordNetLemmatizer object.
    Returns:
        Lemmatized version of the input word.
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    return lemmatizer.lemmatize(word)


def preprocess_line(line):
    return list(line.lower().split(' '))


def fetch_instances(path_to_pos, path_to_neg, verbose=True):
    positive_instances = list()
    if os.path.isdir(path_to_pos):
        for file in os.listdir(os.path.join('train', 'pos')):
            with open(os.path.join('train', 'pos', file), encoding="utf8") as f:
                positive_instances.append(f.read())

    if verbose:
        print("finished reading positive instance files")

    negative_instances = list()
    if os.path.isdir(path_to_neg):
        for file in os.listdir(os.path.join('train', 'neg')):
            with open(os.path.join('train', 'neg', file), encoding="utf8") as f:
                negative_instances.append(f.read())

    if verbose:
        print("finished reading negative instance files")

    # total_instances = positive_instances + negative_instances
    num_docs = len(positive_instances) + len(negative_instances)

    return positive_instances, negative_instances, num_docs


def create_count_matrix(input_list, verbose=True):
    """
    Wrapper around scikit-learn's CountVectorizer class.
    Arguments:
        input_list: list of strings.
    Returns:
        count_feat_mat: feature count csr_matrix.

        count_vectorizer: vectorizer object, see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
    """
    if verbose:
        print(f"launching create_count_matrix")

    count_vectorizer = CountVectorizer(
        input='content',
        encoding='utf-8',
        strip_accents=None,
        lowercase=True,
        # preprocessor=<preprocessor>,
        # tokenizer=<tokenizer>,
        stop_words=None,  # or 'english' or list
        # token_pattern
        # ngram_range
        analyzer=words_and_char_bigrams,
        max_df=1.0,
        min_df=1,
        max_features=None,  # could be int
        # vocabulary: a mapping object, could be useful to map term<->index in output matrix
        binary=False,
        # dtype
    )

    count_feat_mat = count_vectorizer.fit_transform(input_list)
    if verbose:
        print(f"finished create_count_matrix")

    return count_feat_mat, count_vectorizer


def create_tfidf_matrix(input_list, vocabulary_kwarg=None, verbose=True):
    """
    Wrapper around scikit-learn's TfidfVectorizer class.
    Arguments:
        input_list: list of strings.
        vocabulary_kwarg:   optional dictionary argument to match the token<->column index mapping
                            of the output matrix to a specific mapping.
    Returns:
        tfidf_feat_mat: tf-idf csr_matrix.

        tfidf_vectorizer: vectorizer object, see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    if verbose:
        print(f"launching create_tfidf_matrix")

    tfidf_vectorizer = TfidfVectorizer(
        input='content',
        encoding='utf-8',
        strip_accents=None,
        lowercase=True,
        # preprocessor=<preprocessor>,
        # tokenizer=<tokenizer>,
        analyzer="word",
        stop_words=None,  # or 'english' or list
        # token_pattern
        # ngram_range
        max_df=1.0,
        min_df=1,
        max_features=None,  # could be int
        vocabulary=vocabulary_kwarg,  # a mapping object, could be useful to map term<->index in output matrix
        binary=False,
        # dtype
        # norm: 'l1', 'l2', or None
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    )
    tfidf_feat_mat = tfidf_vectorizer.fit_transform(input_list)
    if verbose:
        print(f"finished create_tfidf_matrix")

    return tfidf_feat_mat, tfidf_vectorizer


def main():
    pos_instances_list, neg_instances_list, num_docs = fetch_instances(
        os.path.join('train', 'pos'),
        os.path.join('train', 'neg')
    )

    training_class_labels = np.array([1] * len(pos_instances_list) + [0] * len(neg_instances_list))  # pos = 1, neg = 0

    training_count_feat_mat, training_count_vectorizer = create_count_matrix(pos_instances_list + neg_instances_list)
    token_to_col_index_dict = training_count_vectorizer.vocabulary_

    training_tfidf_feat_mat, training_tfidf_vectorizer = create_tfidf_matrix(pos_instances_list + neg_instances_list,
                                                                             vocabulary_kwarg=token_to_col_index_dict)

    print("pickling")
    with open('training_count_feat_mat_and_vectorizer.pickle', 'wb') as handle:
        pickle.dump((training_count_feat_mat, training_class_labels, training_count_vectorizer), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open('training_tfidf_feat_mat_and_vectorizer.pickle', 'wb') as handle:
        pickle.dump((training_tfidf_feat_mat, training_class_labels, training_tfidf_vectorizer), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print(f"finished")


if __name__ == '__main__':
    main()
