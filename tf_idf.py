import os, operator, random, json, csv, re, nltk, pickle

import numpy as np
import pandas as pds

from scipy.sparse import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer, punkt, sent_tokenize
from tqdm import trange
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('punkt')

NUM_WORDS = 252192 # 300
SHUFFLE_SEED = 4

def loader( filename, pickled_file=False ):

    if pickled_file:
        with open( filename, 'rb' ) as handle:
            matrix, labels, vectorizer = pickle.load( handle )
        return matrix, labels, vectorizer

    else:
        features, labels = [],[]
        with open( filename, 'r', encoding="utf8", newline='\n') as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            for row in csv_reader:
                try:
                    features.append(np.array(json.loads(row[0])))
                    labels.append(np.array(json.loads(row[1])))
                except IndexError:
                    print( row )

        X = np.array(features)
        y = np.array(labels)

        print('done')
        print( X.shape )
        print( y.shape )

        return X,y

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
    
    positive_instances.sort()
    negative_instances.sort()
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

def create_count_matrix( input_list, vocabulary_kwarg=None, verbose=True ):
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
    
    count_vectorizer = None
    if vocabulary_kwarg is None:
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
            ngram_range=(1,2),
            analyzer="word",
            max_df=1.0,
            min_df=1,
            max_features=None, # could be int
            # vocabulary: a mapping object, could be useful to map term<->index in output matrix
            binary=False
            # dtype
        )
    else:
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
            ngram_range=(1,2),
            analyzer="word",
            max_df=1.0,
            min_df=1,
            max_features=None, # could be int
            vocabulary=vocabulary_kwarg, # a mapping object, could be useful to map term<->index in output matrix
            binary=False
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
        ngram_range=(1,2),
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

def main():

    pos_instances_list, neg_instances_list, num_docs = fetch_instances(
        os.path.join( 'train', 'pos' ),
        os.path.join( 'train', 'neg' )
    )

    '''
        This was to select the first and last sentences only, it didn't help.
        #pos_instances_lastline = [ sentence_tokenize( review, ntop=1, reverse=True ) for review in pos_instances_list ]
        #pos_instances_firstline = [ sentence_tokenize( review, ntop=1, reverse=False ) for review in pos_instances_list ]
        #pos_instances_list = [ first+' '+last for ( first, last ) in zip( pos_instances_firstline, pos_instances_lastline) if first != last ]

        #neg_instances_lastline = [ sentence_tokenize( review, ntop=1, reverse=True ) for review in neg_instances_list ]
        #neg_instances_firstline = [ sentence_tokenize( review, ntop=1, reverse=False ) for review in neg_instances_list ]
        #neg_instances_list = [ first+' '+last for ( first, last ) in zip( neg_instances_firstline, neg_instances_lastline ) if first != last ]
    '''

    
    word_count = np.array(
            [ len( word_tokenize( instance ) ) for instance in pos_instances_list+neg_instances_list ]
        )
    
    sentences_count = np.array( 
            [ len( sentence_tokenize( instance, ntop=0 ) ) for instance in pos_instances_list+neg_instances_list ]
         )
    
    avg_word_per_sentences = word_count / sentences_count

    # valence stuff
    wea_valence_file_path = "Ratings_Warriner_et_al-1.csv"
    valence_df = pds.read_csv( wea_valence_file_path, sep=',', usecols=[1,2], index_col=0 )
    mean_valence = np.mean( valence_df["V.Mean.Sum"] )
    valent_scores_list, valent_word_counts_list = valence_score_review( pos_instances_list+neg_instances_list, valence_df, mean_valence, flip_if_not_in_ngram=1 )
    min_valence = abs( min( valent_scores_list ) )
    for i,v in enumerate( valent_scores_list ): # nbsvm assumes the input X is all positive
        valent_scores_list[i] += min_valence


    # positive and negative overlap counts
    # cite http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    with open( os.path.join( 'opinion-lexicon-English', 'negative-words-only.txt' ) , 'r', encoding='utf8' ) as infile:
        negative_words_set = set( [ line.rstrip() for line in infile.readlines() ] )
    with open( os.path.join( 'opinion-lexicon-English', 'positive-words-only.txt' ) , 'r', encoding='utf8' ) as infile:
        positive_words_set = set( [ line.rstrip() for line in infile.readlines() ] )
    
    negative_words = negative_words_set.union( set( [ lemmatize( word ) for word in negative_words_set ] ) )
    
    positive_words = positive_words_set.union( set( [ lemmatize( word ) for word in positive_words_set ] ) )

    positive_word_overlap = [
        len( 
            positive_words.intersection( 
                set( [ lemmatize( word ) for word in word_tokenize( review ) ] ) 
            ) 
        ) for review in pos_instances_list+neg_instances_list
    ]

    negative_word_overlap = [
        len( 
            negative_words.intersection( 
                set( [ lemmatize( word ) for word in word_tokenize( review ) ] ) 
            ) 
        ) for review in pos_instances_list+neg_instances_list
    ]

    # converting to csr_matrices
    word_count = csr_matrix( word_count ).transpose()
    sentences_count = csr_matrix( sentences_count ).transpose()
    avg_word_per_sentences = csr_matrix( avg_word_per_sentences ).transpose()
    valent_scores = csr_matrix( valent_scores_list ).transpose()
    valent_word_counts = csr_matrix( valent_word_counts_list ).transpose()
    valency_ratio_list = []
    for ( vs,vwc ) in zip( valent_scores_list, valent_word_counts_list ):
        if vwc == 0:
            valency_ratio_list.append(0.0)
        else:
            valency_ratio_list.append( vs/ vwc )
    valency_ratio = csr_matrix( 
        valency_ratio_list
    ).transpose()
    print(f"shape of valency_ratio_list = {valency_ratio.shape}")
    positive_word_overlap_csr = csr_matrix(
        np.array( positive_word_overlap )
    ).transpose()
    negative_word_overlap_csr = csr_matrix(
        np.array( negative_word_overlap )
    ).transpose()




    training_class_labels = np.array( [1]*len( pos_instances_list ) + [0]*len( neg_instances_list ) ) # pos = 1, neg = 0

    training_count_feat_mat, training_count_vectorizer = create_count_matrix( pos_instances_list+neg_instances_list )
    token_to_col_index_dict = training_count_vectorizer.vocabulary_ 

    training_tfidf_feat_mat, training_tfidf_vectorizer = create_tfidf_matrix( pos_instances_list+neg_instances_list, vocabulary_kwarg=token_to_col_index_dict )


    ### Add features here ###
    training_count_feat_mat = csr_matrix(
        hstack(
            [ training_count_feat_mat, 
            word_count, 
            sentences_count, 
            avg_word_per_sentences,
            valent_scores,
            valent_word_counts,
            valency_ratio,
            positive_word_overlap_csr,
            negative_word_overlap_csr ]
        )
    )

    training_tfidf_feat_mat = csr_matrix(
        hstack(
            [ training_tfidf_feat_mat, 
            word_count, 
            sentences_count, 
            avg_word_per_sentences,
            valent_scores,
            valent_word_counts,
            valency_ratio,
            positive_word_overlap_csr,
            negative_word_overlap_csr ]
        )
    )

    print("pickling")

    metadata={
        "review":"entire",
        "n-grams":"(1,2)",
        "additional features":"word count, sentences count, avg word/sentence, valency score (wea, min_char_per_words=1, flip_=1), valent word count, valency ratio, positive word overlap, negative word overlap",
        "tokenizer":"homebrew"
    }

    with open( 'entire_posneg_overlap_wea-valency_(1-2)_wc_sc_wps_homebrew_count_feat_mat_labels_and_vectorizer.pickle', 'wb' ) as handle:
        pickle.dump( ( training_count_feat_mat, training_class_labels, training_count_vectorizer, {"pos_instances_list":pos_instances_list, "neg_instances_list":neg_instances_list}, metadata ), handle, protocol=pickle.HIGHEST_PROTOCOL )
    
    with open( 'entire_posneg_overlap_wea-valency_(1-2)_wc_sc_wps_homebrew_tfidf_feat_mat_labels_and_vectorizer.pickle', 'wb' ) as handle:
        pickle.dump( ( training_tfidf_feat_mat, training_class_labels, training_tfidf_vectorizer, {"pos_instances_list":pos_instances_list, "neg_instances_list":neg_instances_list}, metadata ), handle, protocol=pickle.HIGHEST_PROTOCOL )
    
    print(f"finished")

if __name__ == '__main__':
    main()