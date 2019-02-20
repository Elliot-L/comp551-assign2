# standard lib dependencies
import os, operator, random, json, csv, re, pickle, inspect
import numpy as np
import pandas as pds

from tqdm import trange
from collections import Counter
from math import log, e, inf
from more_itertools import consecutive_groups

# scipy dependencies
from scipy.sparse import csr_matrix, hstack

# nltk dependencies
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer, punkt, sent_tokenize
nltk.download('punkt')

# scikit-learn dependencies
from sklearn import datasets, tree
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# external dependencies
from nbsvm import NBSVM

def kfold_plotter( training_indices, testing_indices ):

    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=1 )
    bar_width=0.4
    bar_positions = np.arange( len( training_indices ) )
    for i, ( cv_train, cv_test ) in enumerate( zip( training_indices, testing_indices ) ):
        cv_train += 1
        cv_test += 1
        #print( cv_train )
        X_train_groups = [ list( group ) for group in consecutive_groups( cv_train ) ]
        X_test_groups = [ list( group ) for group in consecutive_groups( cv_test ) ]
        #print( X_train_groups )
        #print( X_test_groups )
        x_train_test_ranges = X_train_groups + X_test_groups
        x_train_test_ranges.sort( key=lambda x:x[0] )
        for x_range in x_train_test_ranges:
            ax.bar( bar_positions[i], ( ( x_range[-1] + 1 ) - x_range[0] ), bar_width, bottom=( x_range[0] - 1 ) )
    plt.show()

def k_fold(X,y, k=5, plot=False ):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    kf = KFold(n_splits=k)
    training_indices, testing_indices = [],[]
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        training_indices.append( train_index )
        testing_indices.append( test_index )
    
    if plot:
        kfold_plotter( training_indices, testing_indices )

    return X_train_list,X_test_list,y_train_list,y_test_list

def strat_k_fold(X,y, k=5, plot=False):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    kf = StratifiedKFold(n_splits=k)
    training_indices, testing_indices = [],[]
    for train_index, test_index in kf.split(X,y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(X_train)
        #print(y_train)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        training_indices.append( train_index )
        testing_indices.append( test_index )
    
    if plot:
        kfold_plotter( training_indices, testing_indices )

    return X_train_list, X_test_list, y_train_list, y_test_list

def fetch_instances( path_to_pos_training_dir, path_to_neg_training_dir, path_to_test_dir, verbose=True ):
    """
    Loads and returns the training and testing corpora.

    Arguments:

        path_to_pos_training_dir: path to the directory containing the positive review files used for training.

        path_to_neg_training_dir: path to the directory containing the negative review files used for training.

        path_to_test_dir: path to the directory containing the reviews used for testing.

        verbose: boolean indicator of verbosity.

    Returns:

            positive_instances: a list of strings, where a positive training review's contents are contained in a single string.

            negative_instances: a list of strings, where a negative training review's contents are contained in a single string.

            num_training_reviews: the number of positive and negative training reviews.
            
            testing_instances: a list of strings, where a testing review's contents are contained in a single string.

    """
    # load positive training instances
    positive_instances = list()
    if os.path.isdir( path_to_pos_training_dir ):
        for file in os.listdir( os.path.join( 'train', 'pos' ) ):
            with open( os.path.join( 'train', 'pos', file ), encoding="utf8" ) as f:
                positive_instances.append( f.read() )
    
    if verbose:
        print( f"finished reading {len( positive_instances )} positive instance files" )

    # load negative training instances
    negative_instances = list()
    if os.path.isdir( path_to_neg_training_dir ):
        for file in os.listdir( os.path.join( 'train', 'neg' ) ):
            with open( os.path.join( 'train', 'neg', file ), encoding="utf8" ) as f:
                negative_instances.append( f.read() )

    if verbose:
        print( f"finished reading {len( negative_instances )} negative instance files" )

    testing_instances = list()
    test_file_names = list()
    if os.path.isdir( path_to_test_dir ):
        for file in os.listdir( path_to_test_dir ):
            with open( os.path.join( path_to_test_dir, file ), encoding="utf8" ) as f:
                testing_instances.append( f.read() )
                test_file_names.append( os.path.basename( file ) )

    assert len( testing_instances ) == len( test_file_names )
    if verbose:
        print( f"finished reading {len( testing_instances )} testing instance files" )
    
    num_training_reviews = len( positive_instances ) + len( negative_instances )
    
    positive_instances.sort() # force a consistent ordering - useful in debugging
    negative_instances.sort() # force a consistent ordering - useful in debugging

    return positive_instances, negative_instances, num_training_reviews, testing_instances, test_file_names
    
def apply_warriner_valence( path_to_warriner_valence_file, instances ):
    """
    Loads Warriner et al.'s "word valence" data and computes the "valence score" and "valent word count" for each instance.
    
    Arguments:

        path_to_warriner_valence_file: path to the valence csv file.

        instances: list of reviews (each review being a single string) for which we want to calculate the warriner valence score and valent-word count.
    
    Returns:

        valent_scores_list: list of the "valence score" of each review in "instances" (in the same order) as computed by valence_score_review.

        valent_word_counts_list: list of the number of "valent" words in each review in "instances" (in the same order) as computed by valence_score_review.

    """
    warriner_valence_df = pds.read_csv( path_to_warriner_valence_file, sep=',', usecols=[1,2], index_col=0 )
    mean_valence = np.mean( warriner_valence_df["V.Mean.Sum"] )
    valent_scores_list, valent_word_counts_list = valence_score_review( instances, warriner_valence_df, mean_valence, flip_if_not_in_ngram=1 )
    min_valence = abs( min( valent_scores_list ) )

    for i,_ in enumerate( valent_scores_list ): # nbsvm assumes the input X is all positive
        valent_scores_list[i] += min_valence

    return valent_scores_list, valent_word_counts_list

def valence_score_review( reviews, valence_df:pds.DataFrame, valence_mean:float, min_chars_per_word=1, flip_if_not_in_ngram=1 ):
    """

    Arguments:

        reviews: list of strings (the raw text reviews).

        valence_df: pandas DataFrame corresponding to the first two columns of Warriner et al's excel sheet.

        valence_mean: (float) the mean valence across the valence_df, used to convert the scores from all + to +/-.

        min_chars_per_word: (int) minimum number of valid characters in a word (to filter out single-character words and punctuation characters).

        flip_if_not_in_ngram: integer indicating how many preceding valid words need to be examined for "not" and "no".

    Returns:

        scores: in-order valence score for each review.

    """
    print( np.mean( valence_df["V.Mean.Sum"] ) )
    valence_df["V.Mean.Sum"] = valence_df["V.Mean.Sum"].subtract( valence_mean )
    print( np.mean( valence_df["V.Mean.Sum"] ) )

    scores = []
    valent_word_counts = []
    for review_index in trange( len( reviews ) ):
        review = reviews[ review_index ]
        score = 0.0
        valent_words = 0
        decontracted_review = decontract( review )
        sentences = sentence_tokenize( decontracted_review, output='list' )
        for sentence in sentences:
            lemmas = [ lemmatize( word ) for word in word_tokenize( sentence ) if len( word ) > min_chars_per_word ]
            for lemma_index, lemma in enumerate( lemmas ):
                try:
                    
                    if flip_if_not_in_ngram > 1:
                        preceding_nots_nos = sum( 
                            [ 1 for preceding_index in range( -flip_if_not_in_ngram, 0 ) 
                            if (
                                ( lemma_index + preceding_index >= 0 ) and 
                                ( lemmas[ lemma_index+preceding_index ] == 'no' or lemmas[ preceding_index ] == 'not' )
                            ) ] 
                        )
                        '''
                        this might not help...
                        subsequent_nots_nos = sum(
                            [ 1 for subsequent_index in range( 1,flip_if_not_in_ngram+1 ) 
                            if lemmas[ subsequent_index ] == 'no' or lemmas[ subsequent_index ] == 'not' ] 
                        )
                        '''
                        score += float( valence_df.loc[lemma, "V.Mean.Sum"] ) * ( (-1)**preceding_nots_nos ) 
                        valent_words += 1
                    else:
                        score += float( valence_df.loc[lemma, "V.Mean.Sum"] ) 
                        valent_words += 1
                except KeyError: # if lemma is not in valence dataframe
                    continue
        scores.append( score )
        valent_word_counts.append( valent_words )

    return scores, valent_word_counts

def has_some_alphanumeric_characters( line ):
    """
    Re wrapper returning True/False depending on whether the input line (string) contains at least one alphabetic character.
    
    Input:
        
        line: a string.
        
    Returns:
    
        boolean True/False.
        
    """
    # to handle misshappen ellipses:
    formatted_line = re.sub( r'\...', r'…', line )
    if re.search('[a-zA-Z!?…]', formatted_line): # hotfixed this, it was previously if re.search('[a-zA-Z!?…]',line):
        return True
    else:
        return False

def sentence_tokenize( text: str, ntop=0, output='str', reverse=False ):
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
            if output == 'list':
                return list( reversed( sentences ) )[:ntop]
            else:
                return ' '.join( list( reversed( sentences ) )[:ntop] )
            
        else:
            if output == 'list':
                return list( reversed( sentences ) )
            else:
                return ' '.join( sentences[::-1] )
    else:
        if ntop > 0:
            if output == 'list':
                return list( sentences )[:ntop]
            else:
                return ' '.join( sentences[:ntop] )
        else:
            if output == 'list':
                return list( sentences )
            else:
                return ' '.join( sentences )

def word_tokenize( line: str, method='homebrew', verbose=False ):
    """
    Tokenizes input string into words (where each word has >= 1 letter).
    
    Input:
        
        line: string to tokenize into words.
            
        method: string indicating which tokenizing method to use (can be 'nltk.word_tokenize', 'WordPunctTokenizer', 'pos', or 'homebrew').

        verbose: boolean indicator of verbosity. 

    Returns: 
        
        a list of all tokens with >= 1 alphabetic character.
        
    """
    assert method in ['pos','nltk.word_tokenize', 'WordPunctTokenizer', 'homebrew']

    if method == 'pos':
        wpt = nltk.WordPunctTokenizer()
        text = wpt.tokenize( line )
        text_tagged = nltk.pos_tag( text )
        new_text = []
        for word in text_tagged:
            new_text.append(word[0] + "/" + word[1])
            doc = ' '.join(new_text)
        return doc
    
    if method == 'WordPunctTokenizer':
        return WordPunctTokenizer().tokenize( line ) 

    elif method == 'nltk.word_tokenize':
        return [ word for word in nltk.word_tokenize( line ) if has_some_alphanumeric_characters( word ) ]

    elif method == 'homebrew':
        
        #if verbose:
        #    print( f"{type( line )} {line}")
        formatted_line = re.sub( r'\.\.\.', r'…', line )
        
        #if verbose: 
        #    print( f"became {formatted_line}")
        
        for ch in ['\?','!','…']:
            formd_line = re.sub( ch, ' {}'.format( ch.strip( '\\' ) ), formatted_line ) # making ?s, !s, and …s 'word tokens'
            formatted_line = formd_line
        return [ word for word in formatted_line.split(' ') if has_some_alphanumeric_characters( word ) ]

def decontract( line: str, 
    contraction_decontraction_list=[( r"n't", r" not" ), ( r"'m", r" am" ), (r"'re", r" are"), (r"'ve", r" have"), (r"'ll", r" will") ] ):
    """
    Re wrapper to expand contractions (e.g. 'would_n't_' -> 'would not').

    Arguments:

        line: string to decontract.

        contraction_decontraction_list: list of tuples where tuple[0] == contraction, tuple[1] == decontraction. 
        
        DISCLAIMER: since "'s" can be decontracted to "is" or "has", it is excluded by default. same goes for "'d" as it can be decontracted to "had" or "would"
    
    Returns:

        input string without contractions.

    """

    for ( contract, decontract ) in contraction_decontraction_list:
        dec_line = re.sub( contract, decontract, line )
        line = dec_line
    return line

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

def overlap_with_Hu_Liu_lexicon( path_to_lexicon_dir, reviews ):
    """
    Loads Hu and Liu's positive and negative english words lexicon (cite http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) and 
    counts how many positive/negative words were in each review in reviews.

    Arguments:

        path_to_lexicon_dir: path to the directory containing the positive words and negative words files.

        reviews: list of strings, where one review's contents are contained in a single string.

    Returns:

        positive_word_overlap: list of the number of positive lemmas (from Hu and Liu's lexicon) were found in each review.

        negative_word_overlap: list of the number of negative lemmas (from Hu and Liu's lexicon) were found in each review.

    """
    # 

    with open( os.path.join( path_to_lexicon_dir, 'negative-words-only.txt' ) , 'r', encoding='utf8' ) as infile:
        negative_words_set = set( [ line.rstrip() for line in infile.readlines() ] )
    with open( os.path.join( path_to_lexicon_dir, 'positive-words-only.txt' ) , 'r', encoding='utf8' ) as infile:
        positive_words_set = set( [ line.rstrip() for line in infile.readlines() ] )
    
    negative_words = negative_words_set.union( set( [ lemmatize( word ) for word in negative_words_set ] ) )
    
    positive_words = positive_words_set.union( set( [ lemmatize( word ) for word in positive_words_set ] ) )

    positive_word_overlap = [
        len( 
            positive_words.intersection( 
                set( [ lemmatize( word ) for word in word_tokenize( review ) ] ) 
            ) 
        ) for review in reviews
    ]

    negative_word_overlap = [
        len( 
            negative_words.intersection( 
                set( [ lemmatize( word ) for word in word_tokenize( review ) ] ) 
            ) 
        ) for review in reviews
    ]

    return positive_word_overlap, negative_word_overlap

def preprocess_and_gen_feat_matrices( pos_instances_list, neg_instances_list, test_instances_list, ngram_range=(1,1), warriner=False, hu_liu_posneg=False, tfidf=False, POStags=False, emp_reg_min=10, to_lowercase=True, remove_stopwords=None, max_num_vector_features=None, verbose=True, decontract_reviews=False ):

    # 0. optional decontraction of some (not all) contractions
    if decontract_reviews:
        if verbose:
            print("decontracting reviews - this may take a while")
        decontracted_pos_instances_list = []
        for ind in trange( len( pos_instances_list ) ):
            decontracted_pos_instances_list.append( decontract( pos_instances_list[ind] ) )
        
        decontracted_neg_instances_list = []
        for ind in trange( len( neg_instances_list ) ):
            decontracted_neg_instances_list.append( decontract( neg_instances_list[ind] ) )

        decontracted_test_instances_list = []
        for ind in trange( len( test_instances_list ) ):
            decontracted_test_instances_list.append( decontract( test_instances_list[ind] ) )

        #decontracted_pos_instances_list = [ decontract( review ) for review in pos_instances_list ]
        #decontracted_neg_instances_list = [ decontract( review ) for review in neg_instances_list ]
        #decontracted_test_instances_list = [ decontract( review ) for review in test_instances_list ]

        del pos_instances_list
        del neg_instances_list
        del test_instances_list

        pos_instances_list = decontracted_pos_instances_list
        neg_instances_list = decontracted_neg_instances_list
        test_instances_list = decontracted_test_instances_list

    
    # 1. basic word and sentence count features #
    ## 1.1 for the training data ##
    train_word_count = np.array(
            [ len( word_tokenize( instance ) ) for instance in pos_instances_list+neg_instances_list ]
        )
    
    train_sentences_count = np.array( 
            [ len( sentence_tokenize( instance, ntop=0 ) ) for instance in pos_instances_list+neg_instances_list ]
         )
    
    train_avg_word_per_sentences = train_word_count / train_sentences_count

    train_word_count_csr, train_sentences_count_csr, train_avg_word_per_sentence_csr = \
        csr_matrix( train_word_count ).transpose(), csr_matrix( train_sentences_count ).transpose(), csr_matrix( train_avg_word_per_sentences ).transpose()
    
    ### delete lists for memory footprint
    del train_word_count
    del train_sentences_count
    del train_avg_word_per_sentences

    ## 1.2 for the testing data ##
    test_word_count = np.array(
            [ len( word_tokenize( instance ) ) for instance in test_instances_list ]
        ) 

    test_sentences_count = np.array( 
            [ len( sentence_tokenize( instance, ntop=0 ) ) for instance in test_instances_list ]
        )

    test_avg_word_per_sentences = test_word_count / test_sentences_count

    test_word_count_csr, test_sentences_count_csr, test_avg_word_per_sentence_csr = \
        csr_matrix( test_word_count ).transpose(), csr_matrix( test_sentences_count ).transpose(), csr_matrix( test_avg_word_per_sentences ).transpose()

    ### delete lists for memory footprint
    del test_word_count
    del test_sentences_count
    del test_avg_word_per_sentences

    # 2. additional features from external sources #
    ## 2.1 computing and storing warriner valence in train/test valent_scores_csr and valent_word_counts_csr ##
    train_valent_scores_csr, train_valent_word_counts_csr, train_valency_ratio_csr = None, None, None
    test_valent_scores_csr, test_valent_word_counts_csr, test_valency_ratio_csr = None, None, None
    if warriner:
        if verbose:
            print( "computing valence score and valent word count for each review")
        train_valent_scores_list, train_valent_word_counts_list = apply_warriner_valence( "Ratings_Warriner_et_al-1.csv", pos_instances_list+neg_instances_list )
        train_valency_ratio_list = []
        for ( vs,vwc ) in zip( train_valent_scores_list, train_valent_word_counts_list ):
            if vwc == 0:
                train_valency_ratio_list.append( 0.0 )
            else:
                train_valency_ratio_list.append( vs/ vwc )
        
        test_valent_scores_list, test_valent_word_counts_list = apply_warriner_valence( "Ratings_Warriner_et_al-1.csv", test_instances_list )
        test_valency_ratio_list = []
        for ( vs,vwc ) in zip( test_valent_scores_list, test_valent_word_counts_list ):
            if vwc == 0:
                test_valency_ratio_list.append( 0.0 )
            else:
                test_valency_ratio_list.append( vs/ vwc )

        ### convert lists -> csr_matrices
        train_valent_scores_csr, train_valent_word_counts_csr, train_valency_ratio_csr = \
            csr_matrix( train_valent_scores_list ).transpose() , csr_matrix( train_valent_word_counts_list ).transpose(), csr_matrix( train_valency_ratio_list ).transpose()
        test_valent_scores_csr, test_valent_word_counts_csr, test_valency_ratio_csr = \
            csr_matrix( test_valent_scores_list ).transpose(), csr_matrix( test_valent_word_counts_list ).transpose(), csr_matrix( test_valency_ratio_list ).transpose()

        ### delete lists for memory footprint
        del train_valent_scores_list
        del train_valent_word_counts_list
        del test_valent_scores_list
        del test_valent_word_counts_list
        del train_valency_ratio_list
        del test_valency_ratio_list

    ## 2.2 computing and storing hu liu positive and negative english lexicon overlaps in train/test hu_liu_positive/negative_word_overlap_csr ##
    train_hu_liu_positive_word_overlap_csr, train_hu_liu_negative_word_overlap_csr = None, None
    test_hu_liu_positive_word_overlap_csr, test_hu_liu_negative_word_overlap_csr = None, None
    if hu_liu_posneg:
        if verbose:
            print( "computing positive-word and negative-word overlaps for each review")
        hu_liu_positive_word_overlap_list, hu_liu_negative_word_overlap_list = overlap_with_Hu_Liu_lexicon( os.path.join( 'opinion-lexicon-English' ), pos_instances_list+neg_instances_list )
        test_hu_liu_positive_word_overlap_list, test_hu_liu_negative_word_overlap_list = overlap_with_Hu_Liu_lexicon( os.path.join( 'opinion-lexicon-English' ), test_instances_list )

        ### convert lists -> csr_matrices
        train_hu_liu_positive_word_overlap_csr, train_hu_liu_negative_word_overlap_csr = \
            csr_matrix( hu_liu_positive_word_overlap_list ).transpose(), csr_matrix( hu_liu_negative_word_overlap_list ).transpose()

        test_hu_liu_positive_word_overlap_csr, test_hu_liu_negative_word_overlap_csr = \
            csr_matrix( test_hu_liu_positive_word_overlap_list ).transpose(), csr_matrix( test_hu_liu_negative_word_overlap_list ).transpose()
        
        ### delete lists for memory footprint
        del hu_liu_positive_word_overlap_list
        del hu_liu_negative_word_overlap_list
        del test_hu_liu_positive_word_overlap_list
        del test_hu_liu_negative_word_overlap_list
    
    # 3. POS tagging #
    if POStags:
        if verbose:
            print("POS-tagging reviews - this may take a while")
        POStagged_pos_instances_list = []
        for ind in trange( len( pos_instances_list ) ):
            POStagged_pos_instances_list.append( word_tokenize( pos_instances_list[ind], method='pos' ) )
        
        POStagged_neg_instances_list = []
        for ind in trange( len( neg_instances_list ) ):
            POStagged_neg_instances_list.append( word_tokenize( neg_instances_list[ind], method='pos' ) )

        POStagged_test_instances_list = []
        for ind in trange( len( test_instances_list ) ):
            POStagged_test_instances_list.append( word_tokenize( test_instances_list[ind], method='pos' ) )

        del pos_instances_list
        del neg_instances_list
        del test_instances_list

        pos_instances_list = POStagged_pos_instances_list
        neg_instances_list = POStagged_neg_instances_list
        test_instances_list = POStagged_test_instances_list

    # 4. compute lexical feature counts/tf-idf #
    ## 4.1 instantiate vectorizer ##
    training_text_vectorizer = None
    
    if tfidf:
        if verbose:
            print( "making tfidf text vectorizer" )
        training_text_vectorizer = TfidfVectorizer(
            input='content',
            encoding='utf-8',
            strip_accents=None,
            lowercase=to_lowercase, 
            # preprocessor=<preprocessor>,
            tokenizer=CustomTokenizer(),
            analyzer="word",
            stop_words=None, # or 'english' or list
            token_pattern=None,
            ngram_range=ngram_range,
            max_df=1.0,
            min_df=emp_reg_min,
            max_features=max_num_vector_features, 
            #vocabulary: a mapping object, could be useful to map term<->index in output matrix
            binary=False,
            # dtype
            # norm: 'l1', 'l2', or None
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False
        )
    else:
        if verbose:
            print( "making count text vectorizer" )
        training_text_vectorizer = CountVectorizer(
            input='content',
            encoding='utf-8',
            strip_accents=None,
            lowercase=to_lowercase, 
            # preprocessor=<preprocessor>,
            tokenizer=CustomTokenizer(),
            stop_words=None, # or 'english' or list
            token_pattern=None,
            ngram_range=ngram_range,
            analyzer="word",
            max_df=1.0,
            min_df=emp_reg_min,
            max_features=max_num_vector_features, 
            # vocabulary: a mapping object, could be useful to map term<->index in output matrix
            binary=False
            # dtype
        )
    
    ## 4.2 vectorize training instances ##
    if verbose:
        print( "vectorizing training reviews" )
    training_count_feature_matrix = training_text_vectorizer.fit_transform( pos_instances_list+neg_instances_list )

    ## 4.3 vectorize testing instances using same vocabulary as for the training instances' vectorizer ##
    testing_text_vectorizer = clone( training_text_vectorizer )
    testing_text_vectorizer.set_params( vocabulary=training_text_vectorizer.vocabulary_ )
    if verbose:
        print( "vectorizing testing reviews" )
    testing_count_feature_matrix = testing_text_vectorizer.fit_transform( test_instances_list )

    # 5. combine count/tf-idf feature matrix with additional features #
    if verbose:
        print( "combining features into a single feature matrix" )
    training_feature_matrix = csr_matrix(
        hstack(
            [ training_count_feature_matrix, 
            train_word_count_csr, train_sentences_count_csr, train_avg_word_per_sentence_csr, 
            train_valent_scores_csr, train_valent_word_counts_csr, train_valency_ratio_csr,
            train_hu_liu_positive_word_overlap_csr, train_hu_liu_negative_word_overlap_csr,
            ]
        )
    )

    testing_feature_matrix = csr_matrix(
        hstack(
            [
                testing_count_feature_matrix,
                test_word_count_csr, test_sentences_count_csr, test_avg_word_per_sentence_csr,
                test_valent_scores_csr, test_valent_word_counts_csr, test_valency_ratio_csr,
                test_hu_liu_positive_word_overlap_csr, test_hu_liu_negative_word_overlap_csr
            ]
        )
    )

    funct_frame_for_metadata = inspect.currentframe()
    funct_args, _, _, funct_values = inspect.getargvalues( funct_frame_for_metadata )
    funct_arguments = [ (i, funct_values[i]) for i in funct_args ]
    
    metadata = {
        'preprocessing arguments': funct_arguments,
        'training text vectorizer arguments': training_text_vectorizer.get_params(),
        'testing text vectorizer arguments': testing_text_vectorizer.get_params()
    }

    training_labels_list = [1]*len( pos_instances_list ) + [0]*len( neg_instances_list )
    training_labels = np.array( training_labels_list )
    if verbose:
        print("finished preprocessing -> feature matrix generation")

    return training_feature_matrix, training_labels, testing_feature_matrix, metadata

def run_kfold_experiment( training_feature_matrix, training_labels, folds=5, verbose=True ):

    assert training_feature_matrix.shape[0] == len( training_labels )
    bernoulli_Pipeline = Pipeline([ 
        #( 'TruncatedSVD', TruncatedSVD( n_components=20 ) ),
        ( 'bernoulliNB clf', BernoulliNB(
            alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True ) 
        )
    ])


    logreg_pipeline = Pipeline([   
        #( 'TruncatedSVD', TruncatedSVD( n_components=20 ) ),
        ( 'logreg clf', LogisticRegression(
            random_state=0, solver='liblinear', multi_class='ovr', max_iter=1000
            ) 
        ),  
    ])

    rf_pipeline = Pipeline([  
        #( 'TruncatedSVD', TruncatedSVD( n_components=20 ) ),
        ( 'random forrest clf', RandomForestClassifier() )
    ])

    linscv_pipeline = Pipeline([
        ( 'linsvc clf', LinearSVC() )
    ])

    multinomial_Pipeline = Pipeline([
        ( 'multinomialNB clf', MultinomialNB() )
    ])

    nbsvm_Pipeline = Pipeline([
         ('nbsvm clf', NBSVM() )
    ])

    gbclf_Pipeline = Pipeline([
        ( 'gbclf', GradientBoostingClassifier() )
    ])

    knn_Pipeline = Pipeline([
        ( 'knn clf', KNeighborsClassifier() )
    ])

    pipelines = [ 
        bernoulli_Pipeline, 
        logreg_pipeline, 
        #  rf_pipeline, # not accurate
        linscv_pipeline, 
        multinomial_Pipeline, 
        nbsvm_Pipeline, 
        # knn_Pipeline # not accurate
        #gbclf_Pipeline # slow af
    ]

    splits = strat_k_fold( training_feature_matrix, training_labels, k=folds )
    num_sets = len(splits[0])

    accuracies = { 
        list( pipeline.named_steps.keys() )[0]: [] for pipeline in pipelines
    }

    for i in range(num_sets):
        train_x = splits[0][i]
        valid_x = splits[1][i]
        train_y = splits[2][i]
        valid_y = splits[3][i]

        print('\n')
        print( f"split {i+1} \ {num_sets}")
        for pipeline in pipelines:
            if verbose:
                print( f"training { list( pipeline.named_steps.keys() )[0]}")
            pipeline.fit( train_x, train_y )
        
        predictions = {}
        for pipeline in pipelines:
            if verbose:
                print( f"predicting with {list( pipeline.named_steps.keys() )[0]}")
            predictions[ list( pipeline.named_steps.keys() )[0] ] = pipeline.predict( valid_x )
            accuracies[ list( pipeline.named_steps.keys() )[0] ].append( predictions[ list( pipeline.named_steps.keys() )[0] ] )
        print("\nresults\n")
        for pipeline_name, predictions in predictions.items():
            print( f"{pipeline_name}: {accuracy_score( valid_y, predictions ) }")
        
    return accuracies

def run_on_test_set( training_feature_matrix, training_labels, testing_feature_matrix, test_file_names, verbose=True, outputfilename="nbsvm_output.txt" ):

    bernoulli_Pipeline = Pipeline([ 
        #( 'TruncatedSVD', TruncatedSVD( n_components=20 ) ),
        ( 'bernoulliNB clf', BernoulliNB(
            alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True ) 
        )
    ])

    logreg_pipeline = Pipeline([   
        #( 'TruncatedSVD', TruncatedSVD( n_components=20 ) ),
        ( 'logreg clf', LogisticRegression(
            random_state=0, solver='liblinear', multi_class='ovr', max_iter=1000
            ) 
        ),  
    ])

    rf_pipeline = Pipeline([  
        #( 'TruncatedSVD', TruncatedSVD( n_components=20 ) ),
        ( 'random forrest clf', RandomForestClassifier() )
    ])

    linscv_pipeline = Pipeline([
        ( 'linsvc clf', LinearSVC() )
    ])

    multinomial_Pipeline = Pipeline([
        ( 'multinomialNB clf', MultinomialNB() )
    ])

    nbsvm_Pipeline = Pipeline([
         ('nbsvm clf', NBSVM() )
    ])

    gbclf_Pipeline = Pipeline([
        ( 'gbclf', GradientBoostingClassifier() )
    ])

    knn_Pipeline = Pipeline([
        ( 'knn clf', KNeighborsClassifier() )
    ])

    if verbose:
        print( f"fitting model + feature matrix")

    nbsvm_Pipeline.fit( training_feature_matrix, training_labels )

    print("making predictions")
    predictions = nbsvm_Pipeline.predict( testing_feature_matrix )

    for ( pred, file ) in zip( predictions, test_file_names ):
        print( f"{file}\t{pred}")
    
    file_numbers = [ int( name.split('.txt')[0] ) for name in test_file_names ]
    results_np = np.zeros( ( len( test_file_names ), 2 ) )
    for index, ( pred, file ) in enumerate( zip( predictions, file_numbers ) ):
        results_np[ index, 0 ] = file
        results_np[ index, 1 ] = pred

    results_df = pds.DataFrame( results_np, columns=["Id", "Category"] )
    results_df.sort_values(by=['Id'])
    results_df.to_csv( outputfilename, sep='\t', index=False, header=True, mode='w' )

def main( outputfilename, validate=True, test=False, pickle_matrices_filename=False ):

    pos_train_instances, neg_train_instances, num_train_instances, test_instances, test_file_names = fetch_instances( os.path.join( 'train', 'pos' ), os.path.join( 'train', 'neg' ), os.path.join( 'test' ) )

    # Arguments:
    # pos_instances_list, 
    # neg_instances_list, 
    # test_instances_list, 
    # ngram_range=(1,1), 
    # warriner=False, 
    # hu_liu_posneg=False, 
    # tfidf=False, 
    # POStags=False, 
    # emp_reg_min=10, 
    # to_lowercase=True, 
    # remove_stopwords=None, 
    # max_num_vector_features=None, 
    # verbose=True, 
    # decontract_reviews=False 
    training_feature_matrix, training_labels, testing_feature_matrix, metadata_for_text_to_matrix = preprocess_and_gen_feat_matrices(
        pos_train_instances,
        neg_train_instances,
        test_instances, 
        ngram_range=(1,2),
        warriner=True,
        hu_liu_posneg=False,
        tfidf=True,
        POStags=True,
        emp_reg_min=10,
        to_lowercase=True, 
        remove_stopwords=None,
        max_num_vector_features=None,  
        verbose=True,
        decontract_reviews=True 
    )

    #print( metadata_for_text_to_matrix['preprocessing arguments'] )

    if validate:
        kfold_validation_results = run_kfold_experiment( training_feature_matrix, training_labels )
    
    if test:
        run_on_test_set( training_feature_matrix, training_labels, testing_feature_matrix, test_file_names, verbose=True, outputfilename=outputfilename )
        print( f"saved the predictions in {outputfilename}\n\n" )

    if pickle_matrices_filename:
        pickle_dictionary = {
            'training feature matrix': training_feature_matrix,
            'training labels': training_labels, 
            'testing feature matrix': testing_feature_matrix, 
            'metadata': metadata_for_text_to_matrix
        }
        with open( pickle_matrices_filename, 'wb' ) as handle:
            pickle.dump( pickle_dictionary, protocol=pickle.HIGHEST_PROTOCOL )
        
if __name__ == '__main__':
    main( 'dummy.txt', validate=True, test=False, pickle_matrices_filename=False )