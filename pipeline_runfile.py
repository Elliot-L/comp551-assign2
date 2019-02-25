# standard lib dependencies
import os, operator, random, json, csv, re, pickle
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

# homemade dependencies
from bernoulli_NB import *
from pipelines import strat_k_fold
from tf_idf import *

# external dependencies
from nbsvm import NBSVM

def make_testing_feature_matrix( path_to_dir, vocabulary_kwarg=None ):

    if path_to_dir is None:
        path_to_dir=os.path.join( '..','comp-551-imbd-sentiment-classification', 'test' )

    testing_instances = []
    testing_instances_names = []
    for file in [ os.path.join( path_to_dir, f ) for f in os.listdir( path_to_dir )  ]:
        with open( file, encoding="utf8" ) as f:
            testing_instances_names.append( str( os.path.basename( file ) ) )
            testing_instances.append( f.read() )

    word_count = np.array(
            [ len( word_tokenize( instance ) ) for instance in testing_instances ]
        )
    
    sentences_count = np.array( 
            [ len( sentence_tokenize( instance, ntop=0 ) ) for instance in testing_instances ]
        )
    
    avg_word_per_sentences = word_count / sentences_count

    # valence stuff
    wea_valence_file_path = "Ratings_Warriner_et_al-1.csv"
    valence_df = pds.read_csv( wea_valence_file_path, sep=',', usecols=[1,2], index_col=0 )
    mean_valence = np.mean( valence_df["V.Mean.Sum"] )
    valent_scores_list, valent_word_counts_list = valence_score_review( testing_instances, valence_df, mean_valence, flip_if_not_in_ngram=1 )
    min_valence = abs( min( valent_scores_list ) )
    for i,v in enumerate( valent_scores_list ): # nbsvm assumes the input X is all positive
        valent_scores_list[i] += min_valence

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
            valency_ratio_list.append( vs / vwc )
    valency_ratio = csr_matrix( 
        valency_ratio_list
    ).transpose()
    print(f"shape of valency_ratio_list = {valency_ratio.shape}")
    # not used
    ##positive_word_overlap_csr = csr_matrix(
    ##    np.array( positive_word_overlap )
    ##).transpose()
    ##negative_word_overlap_csr = csr_matrix(
    #3    np.array( negative_word_overlap )
    #3).transpose()

    testing_count_feat_mat, testing_count_vectorizer = create_count_matrix( testing_instances, vocabulary_kwarg=vocabulary_kwarg )
    token_to_col_index_dict = testing_count_vectorizer.vocabulary_ 

    # training_tfidf_feat_mat, training_tfidf_vectorizer = create_tfidf_matrix( pos_instances_list+neg_instances_list, vocabulary_kwarg=token_to_col_index_dict )


    ### Add features here ###
    testing_count_feat_mat = csr_matrix(
        hstack(
            [ testing_count_feat_mat, 
            word_count, 
            sentences_count, 
            avg_word_per_sentences,
            valent_scores,
            valent_word_counts,
            valency_ratio#,
            #positive_word_overlap_csr,
            #negative_word_overlap_csr 
            ]
        )
    )

    return testing_count_feat_mat, testing_instances_names


def iris_logreg():

    logreg_pipeline = Pipeline([

        ( 'logreg_clf', LogisticRegression(
            random_state=0, solver='liblinear', multi_class='ovr'
            ) 
        ), 
        
    ])

    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, test_size=0.5 )

    logreg_pipeline.fit( X_train, y_train )
    y_pred = logreg_pipeline.predict( X_test )
    print( accuracy_score( y_test, y_pred ) )
    print( logreg_pipeline )

def run_pipelines( verbose=True ):

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

    # X, y, vectorizer = loader( "unigram_homebrew_tokenized_count_feat_mat_labels_and_vectorizer.pickle", pickled_file=True )
    with open( 'entire_wea-valency_(1-2)_wc_sc_wps_homebrew_count_feat_mat_labels_and_vectorizer.pickle', 'rb' ) as handle:
        try:
            X, y, vectorizer, mapping, metadata = pickle.load( handle )
        except ValueError:
            X, y, vectorizer = pickle.load( handle )

    #trunc_SVD = TruncatedSVD( n_components=50 )
    #trunc_SVD.fit( X )

    

    '''
    for training on the whole dataset.
    '''
    
    print("starting training")
    nbsvm_Pipeline.fit( X, y )

    print("making test feature matrix")
    testing_feature_matrix, testing_filenames = make_testing_feature_matrix( 
        os.path.join( '..','comp-551-imbd-sentiment-classification', 'test' ), vocabulary_kwarg=vectorizer.vocabulary_  
    )
    print("making predictions")
    predictions = nbsvm_Pipeline.predict( testing_feature_matrix )

    for ( pred, file ) in zip( predictions, testing_filenames ):
        print( f"{file}\t{pred}")
    
    file_numbers = [ int( name.split('.txt')[0] ) for name in testing_filenames ]
    results_np = np.zeros( ( len( testing_filenames ), 2 ) )
    for index, ( pred, file ) in enumerate( zip( predictions, file_numbers ) ):
        results_np[ index, 0 ] = file
        results_np[ index, 1 ] = pred

    results_df = pds.DataFrame( results_np, columns=["Id", "Category"] )
    results_df.sort_values(by=['Id'])
    results_df.to_csv( "results.tsv", sep='\t', index=False, header=True, mode='w' )
    
    with open("output.txt", "w") as outfile:
        outfile.write("Id   Category\n")
        for ( pred, file ) in zip( predictions, testing_filenames ):
            outfile.write( f"{file}\t{pred}\n") 
    '''
    splits = strat_k_fold(X, y, k=5)
    num_sets = len(splits[0])
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
        
        print("\nresults\n")
        for pipeline_name, predictions in predictions.items():
            print( f"{pipeline_name}: {accuracy_score( valid_y, predictions ) }")
            
    '''

    print("done")

if __name__ == '__main__':
    run_pipelines()