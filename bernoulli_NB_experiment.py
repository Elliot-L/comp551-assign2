import csv, json, pickle
import numpy as np

from math import log, e, inf
from sklearn.naive_bayes import BernoulliNB
from bernoulli_NB import homemade_BernoulliNB
from pipelines import strat_k_fold
from tf_idf import * 

def run_experiment( X_float, y_float ):
    """
    Function to use the homemade bernoulli_NB classifier on the feature matrix and target/class matrix provided.

    Arguments:

        X_float: | instances | x | features | -shaped numpy array ( of dtype = float ) representing the feature matrix.

        y_float: | instances | x 1 -shaped numpy vector representing the target labels.
    """
    X = X_float.astype( bool )
    y = ( y_float > 0 ).astype( int )
    
    splits = strat_k_fold(X, y, k=5) 
    num_sets = len(splits[0])
    accuracies = []
    sanity_accuracies = []
    for i in range(num_sets):
        train_x = splits[0][i]
        valid_x = splits[1][i]
        train_y = splits[2][i]
        valid_y = splits[3][i]

        # homemade_BernoulliNB
        bernie = homemade_BernoulliNB()
        bernie.fit( train_x, train_y )
        # sanity checks
        sanity_check = BernoulliNB()
        sanity_check.fit( train_x, train_y )

        predictions = bernie.predict( valid_x )
        sanity_preds = sanity_check.predict( valid_x )

        # compute accuracy
        num_correct = 0.0
        for i in range(len(predictions)):
            if predictions[i] == valid_y[i]:
                num_correct += 1.0
        set_accuracy = num_correct / float(len(predictions))

        sanity_num_correct = 0.0
        for i in range(len(sanity_preds)):
            if sanity_preds[i] == valid_y[i]:
                sanity_num_correct += 1.0
        sanity_set_accuracy = sanity_num_correct / float( len( sanity_preds))

        print("Set accuracy: " + str(set_accuracy))
        print( f"Sanity accuracy: {sanity_set_accuracy}" )
        sanity_accuracies.append( sanity_set_accuracy )
        accuracies.append(set_accuracy)
    
    print("Total accuracy: " + str(np.average(accuracies)))
    print(f"Total sanity accuracy: {np.average(sanity_accuracies)}")

def main():
    X, y, vectorizer = loader( "unigram_homebrew_tokenized_tfidf_feat_mat_labels_and_vectorizer.pickle", pickled_file=True )
    run_experiment( X, y )

if __name__ == '__main__':
    main()