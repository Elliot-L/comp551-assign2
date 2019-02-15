import os, operator, random, json, csv, re, nltk, pickle

import numpy as np

from scipy.sparse import csr_matrix
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer, punkt, sent_tokenize
from tqdm import trange
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('punkt')

NUM_WORDS = 252192 # 300
SHUFFLE_SEED = 4

def has_some_alphanumeric_characters( line ):
    """
    Re wrapper returning True/False depending on whether the input line (string) contains at least one alphabetic character.
    
    Input:
        
        line: a string.
        
    Returns:
    
        boolean True/False
        
    """
    # to handle misshappen ellipses:
    formatted_line = re.sub( r'\...', r'…', line )
    if re.search('[a-zA-Z!?…]', line):
        return True
    else:
        return False

def sentence_tokenize( text: str, ntop=0, reverse=False ):
    """
    Wrapper around NLTK's sent_tokenize English sentence tokenizer.

    Arguments:

        text: text to tokenize into sentences. 

        ntop: specifies how many sentences the function will return.

        reverse: boolean indicator of whether the sentences should be returned in their original or reverse order.

    Returns:

        The ntop first/last sentences in the input text.

    """
    sentences = sent_tokenize( text )
    if reverse:
        if ntop > 0: 
            return list( reversed( sentences ) )[:ntop]
        else:
            return sentences[::-1]
    else:
        if ntop > 0:
            return sentences[:ntop]
        else:
            return sentences

def word_tokenize( line: str, method='homebrew', verbose=False ):
    """
    Tokenizes input string into words (where each word has >= 1 letter).
    
    Input:
        
        line: string to tokenize into words.
            
        method: string indicating which tokenizing method to use (can be 'nltk.word_tokenize', 'WordPunctTokenizer', or 'homebrew').

        verbose: boolean indicator of verbosity. 

    Returns: 
        
        a list of all tokens with >= 1 alphabetic character.
        
    """
    assert method in ['nltk.word_tokenize', 'WordPunctTokenizer', 'homebrew']
    
    if method == 'WordPunctTokenizer':
        return WordPunctTokenizer().tokenize( line ) 

    elif method == 'nltk.word_tokenize':
        return [ word for word in nltk.word_tokenize( line ) if has_some_alphanumeric_characters( word ) ]

    elif method == 'homebrew':
        
        if verbose:
            print( f"{type( line )} {line}")
        formatted_line = re.sub( r'\.\.\.', r'…', line )
        
        if verbose: 
            print( f"became {formatted_line}")
        
        for ch in ['\?','!','…']:
            formd_line = re.sub( ch, ' {}'.format( ch.strip( '\\' ) ), formatted_line ) # making ?s, !s, and …s 'word tokens'
            formatted_line = formd_line
        return [ word for word in formatted_line.split(' ') if has_some_alphanumeric_characters( word ) ]

def decontract( line: str, 
    contraction_decontraction_list=[( r"n't", r" not" ), ( r"'m", r" am" ), (r"'re", r" are"), (r"'ve", r" have"), (r"'d", r" had"), (r"'ll", r" will") ] ):
    """
    Re wrapper to expand contractions (e.g. 'would_n't_' -> 'would not').

    Arguments:

        line: string to decontract.

        contraction_decontraction_list: list of tuples where tuple[0] == contraction, tuple[1] == decontraction. since "'s" can be decontracted to "is" or "has", it is excluded by default.
    Returns:

        input string without contractions.

    """
    print( "Don't use me, contractions are hard.\nExiting now." )
    raise SystemExit
    for ( contract, decontract ) in contraction_decontraction_list:
        dec_line = re.sub( contract, decontract, line )
        line = dec_line
    return line

def lemmatize( word: str, lemmatizer=None ):
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
    
    return lemmatizer.lemmatize( word )

def preprocess_line(line):
    return list(line.lower().split(' '))

def fetch_instances( path_to_pos, path_to_neg, verbose=True ):

    positive_instances = list()
    if os.path.isdir( path_to_pos ):
        for file in os.listdir( os.path.join( 'train', 'pos' ) ):
            with open( os.path.join( 'train', 'pos', file ), encoding="utf8" ) as f:
                positive_instances.append( f.read() )
    
    if verbose:
        print("finished reading positive instance files")

    negative_instances = list()
    if os.path.isdir( path_to_neg ):
        for file in os.listdir( os.path.join( 'train', 'neg' ) ):
            with open( os.path.join( 'train', 'neg', file ), encoding="utf8" ) as f:
                negative_instances.append( f.read() )

    if verbose:
        print("finished reading negative instance files")

    # total_instances = positive_instances + negative_instances
    num_docs = len( positive_instances ) + len( negative_instances )

    return positive_instances, negative_instances, num_docs

class CustomTokenizer():
    """
    Wrapper class around the word_tokenize function.
    """
    def __init__( self, tokenizing_funct=word_tokenize ):
        self.funct = tokenizing_funct

    def __call__( self, doc ):
        #print( f"calling {self.funct}, {self.funct.__name__}")
        #print( doc )
        #print( "became" )
        res = self.funct( doc )
        #print( res )
        return res

def create_count_matrix( input_list, verbose=True ):
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
        tokenizer=CustomTokenizer(),
        stop_words=None, # or 'english' or list
        # token_pattern
        token_pattern=None,
        ngram_range=(1,1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None, # could be int
        # vocabulary: a mapping object, could be useful to map term<->index in output matrix
        binary=False,
        # dtype
    )

    count_feat_mat = count_vectorizer.fit_transform( input_list )
    if verbose:
        print(f"finished create_count_matrix")
        
    return count_feat_mat, count_vectorizer

def create_tfidf_matrix( input_list, vocabulary_kwarg=None, verbose=True ):
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
        tokenizer=CustomTokenizer(),
        analyzer="word",
        stop_words=None, # or 'english' or list
        # token_pattern
        token_pattern=None,
        ngram_range=(1,1),
        max_df=1.0,
        min_df=1,
        max_features=None, # could be int
        vocabulary=vocabulary_kwarg, # a mapping object, could be useful to map term<->index in output matrix
        binary=False,
        # dtype
        # norm: 'l1', 'l2', or None
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    )

    tfidf_feat_mat = tfidf_vectorizer.fit_transform( input_list )
    if verbose:
        print(f"finished create_tfidf_matrix")
        
    return tfidf_feat_mat, tfidf_vectorizer



def main():

    pos_instances_list, neg_instances_list, num_docs = fetch_instances(
        os.path.join( 'train', 'pos' ),
        os.path.join( 'train', 'neg' )
    )

    training_class_labels = np.array( [1]*len( pos_instances_list ) + [0]*len( neg_instances_list ) )# pos = 1, neg = 0

    training_count_feat_mat, training_count_vectorizer = create_count_matrix( pos_instances_list+neg_instances_list )
    token_to_col_index_dict = training_count_vectorizer.vocabulary_ 

    training_tfidf_feat_mat, training_tfidf_vectorizer = create_tfidf_matrix( pos_instances_list+neg_instances_list, vocabulary_kwarg=token_to_col_index_dict )

    print("pickling")
    with open( 'unigram_homebrew_tokenized_count_feat_mat_labels_and_vectorizer.pickle', 'wb' ) as handle:
        pickle.dump( ( training_count_feat_mat, training_class_labels, training_count_vectorizer ), handle, protocol=pickle.HIGHEST_PROTOCOL )
    
    with open( 'unigram_homebrew_tokenized_tfidf_feat_mat_labels_and_vectorizer.pickle', 'wb' ) as handle:
        pickle.dump( ( training_tfidf_feat_mat, training_class_labels, training_tfidf_vectorizer ), handle, protocol=pickle.HIGHEST_PROTOCOL )
    
    print(f"finished")

if __name__ == '__main__':
    main()