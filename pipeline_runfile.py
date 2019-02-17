# standard lib dependencies
import os, operator, random, json, csv, re, pickle
import numpy as np

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
from tf_idf import fetch_instances, CustomTokenizer, loader, word_tokenize

# external dependencies
from nbsvm import NBSVM

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
        rf_pipeline, 
        linscv_pipeline, 
        multinomial_Pipeline, 
        nbsvm_Pipeline, 
        knn_Pipeline
        #gbclf_Pipeline 
    ]

    X, y, vectorizer = loader( "unigram_homebrew_tokenized_count_feat_mat_labels_and_vectorizer.pickle", pickled_file=True )
    X = X.astype( bool )
    y = ( y > 0 ).astype( int )
    
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
        '''
        bernoulli_Pipeline.fit( train_x, train_y )
        logreg_pipeline.fit( train_x, train_y )
        rf_pipeline.fit( train_x, train_y )
        linscv_pipeline.fit( train_x, train_y )
        multinomial_Pipeline.fit( train_x, train_y )
        nbsvm_Pipeline.fit( train_x, train_y )
        gbclf_Pipeline.fit( train_x, train_y )
        '''

        predictions = {}
        for pipeline in pipelines:
            if verbose:
                print( f"predicting with {list( pipeline.named_steps.keys() )[0]}")
            predictions[ list( pipeline.named_steps.keys() )[0] ] = pipeline.predict( valid_x )
        
        print("\nresults\n")
        for pipeline_name, predictions in predictions.items():
            print( f"{pipeline_name}: {accuracy_score( valid_y, predictions ) }")
            
        
        '''bernoulli_NB_preds = bernoulli_Pipeline.predict( valid_x )
        logreg_preds = logreg_pipeline.predict( valid_x )
        rf_preds = rf_pipeline.predict( valid_x )
        linscv_preds = linscv_pipeline.predict( valid_x )
        multinomi_NB_preds = multinomial_Pipeline.predict( valid_x )
        nbsvm_preds = nbsvm_Pipeline.predict( valid_x )
        #gbclf_preds = gbclf_Pipeline.predict( valid_x )
        
        
        print('===')
        print( f"bernoulliNB: {accuracy_score( valid_y, bernoulli_NB_preds )}" )
        print( f"logreg: {accuracy_score( valid_y, logreg_preds )}" )
        print( f"randomforrest: {accuracy_score( valid_y, rf_preds )}" )
        print( f"linscv: {accuracy_score( valid_y, linscv_preds )}")
        print( f"multinomialNB: { accuracy_score( valid_y, multinomi_NB_preds )}")
        print( f"nbsvm: { accuracy_score( valid_y, nbsvm_preds )}")
        print( f"gbclf: { accuracy_score( valid_y, gbclf_preds )}")'''
    
    print("done")

if __name__ == '__main__':
    run_pipelines()