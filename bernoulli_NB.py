import numpy as np 
from math import log, e, inf
from sklearn.naive_bayes import BernoulliNB

"""
@SC for shame/blame
"""

def binary_bernoulli_NB( binary_feature_array: np.ndarray, binary_class_vec: np.ndarray, laplace_smoothing=True ):
    """
    Computes the MLEs P(feature_i == 1 | target == 1) in a 2-class (binary classification) context. 

    Arguments:

        binary_feature_array: a | instances | x | features | numpy array.

        binary_class_vec: | instances |-sized binary numpy vector.

        laplace_smoothing: boolean indicating whether to use add-1-smoothing or not.

    Returns:

        three unnamed objects (in-order):

            1. log( prob( class=0 ) )
            2. log( prob( class=1 ) ),
            3. 2 x | features | array whose [r,c] entry represents the log( P( feature c | class r ) )

    """

    # some sanity checks
    if binary_feature_array.dtype != 'bool':
        binary_feature_array = binary_feature_array.astype( bool )
    
    if binary_class_vec.dtype != 'bool':
        binary_class_vec = binary_class_vec.astype( bool )

    # if targets is a (1,#) or (#,1) array instead of a (#,) vector
    if binary_class_vec.ndim > 1:
        binary_class_vec = binary_class_vec.flatten()

    # compute P(target=1) and P(target=0)
    count_target1 = np.sum( binary_class_vec )
    count_target0 = len( binary_class_vec ) - count_target1

    # compute P(feature_i | target=1)
    prob_feat_i_cond_1 = np.array( [ 0.0 ] * binary_feature_array.shape[1] , dtype=np.float64 )
    prob_feat_i_cond_0 = np.array( [ 0.0 ] * binary_feature_array.shape[1] , dtype=np.float64 )

    inverted_binary_class_vec = np.invert( binary_class_vec ) # used for P( y=0 | x ), swaps Trues -> Falses
    for feat_i in range( binary_feature_array.shape[1] ):
        # np.logical_and( np.array( [ True, True, False, False ] ), 
        #                 np.array( [ True, False, True, False ] ) )
        # yields True False False False
        prob_feat_i_cond_1[ feat_i ] = np.sum( np.logical_and( binary_feature_array[:,feat_i], binary_class_vec ) )
        prob_feat_i_cond_0[ feat_i ] = np.sum( np.logical_and( binary_feature_array[:,feat_i], inverted_binary_class_vec ) )

    if laplace_smoothing: 
        prob_feat_i_cond_1 = ( prob_feat_i_cond_1 + 1 ) / ( count_target1 + 2 )
        prob_feat_i_cond_0 = ( prob_feat_i_cond_0 + 1 ) / ( count_target0 + 2 )
        return log( count_target0 / len( binary_class_vec ) ), log( count_target1 / len( binary_class_vec ) ), np.log( np.vstack( ( prob_feat_i_cond_0, prob_feat_i_cond_1 ) ) )
    
    else:
        prob_feat_i_cond_1 = ( prob_feat_i_cond_1 ) / ( count_target1 )
        prob_feat_i_cond_0 = ( prob_feat_i_cond_0 ) / ( count_target0 )
        return log( count_target0 / len( binary_class_vec ) ), log( count_target1 / len( binary_class_vec ) ), np.log( np.vstack( ( prob_feat_i_cond_0, prob_feat_i_cond_1 ) ) )
    
def predict( c_x_f_logprob: np.ndarray, log_class_probs, test_instance: np.ndarray ):
    """
    Returns predicted class for test_instance using MLE/MAP parameters and class probabilities.

    Arguments:

        c_x_f_logprob: | classes | x | features | numpy array whose [r,c] entry represents the log( P( feature c | class r ) ).

        log_class_probs: numpy vector or list of log( class probability ).

        test_instance: | features |-sized numpy vector. 
    
    Returns:

        The predicted class ( 0 / 1 ) for the test_instance.
        
    """
    best_class, best_prob = None, -inf

    for row_index, row_class in enumerate( c_x_f_logprob ):
        not_row_class = np.log( np.ones( row_class.shape ) - np.exp( row_class ) )
        probs = np.vstack( ( not_row_class, row_class ) )
        class_prob = log_class_probs[ row_index ] + sum( [ probs[ tie, tii ] for tii, tie in enumerate( test_instance ) ] )
        if class_prob >= best_prob:
            best_class, best_prob = row_index, class_prob 
    
    return best_class

if __name__ == '__main__':

    # random initialization
    classes =  np.random.randint( 0, 2, ( 100 ) )
    features = np.random.randint( 0, 2, ( 100, 100 ) )

    logp0, logp1, res =  binary_bernoulli_NB( features, classes) 
    
    # sanity check with sklearn's BernoulliNB classifier
    sanity_check = BernoulliNB()
    sanity_check.fit( features, classes )
    for i in range( 100000 ):
        tests = np.random.randint( 0, 2, ( 10, 100 ) )
        my_preds = []
        for t in tests:
            my_preds.append( predict( res, [ logp0, logp1 ], t ) )
        
        sanity_preds = sanity_check.predict( tests )
        print( i )
        assert ( np.array_equal( np.array( my_preds ), sanity_preds ) )

