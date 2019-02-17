import numpy as np 
from math import log, e, inf
from sklearn.naive_bayes import BernoulliNB
from collections import Counter
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import trange
"""
@SC for shame/blame
"""

class homemade_BernoulliNB():

    def __init__( self ):
        """
        Class constructor.
        """
        # dictionary of ( 1 : log( frequency of class= 1 ) (key : value pairs)
        self.class_log_probs = dict() 
        # | classes | x | features | array whose [r,c] entry represents the log( P( feature c | class r ) )
        self.learned_logprob_feature_given_class = None 
        
    
    def fit( self, binary_feature_array, binary_class_vec: np.ndarray, laplace_smoothing=True, verbose=True ):
        """
        Computes the MLEs P(feature_i == 1 | target == 1) and the class probabilities in a 2-class (binary classification) context. 

        Arguments:

            binary_feature_array: a | instances | x | features | numpy array or csr_matrix (gets converted to a binary csr_matrix either way).

            binary_class_vec: | instances |-sized binary numpy vector.

            laplace_smoothing: boolean indicating whether to use add-1-smoothing or not.

            verbose: boolean indicator of verbosity.

        Returns:

            Nothing.

        Updates:

            self.class_log_probs: initializes ( <class int> : log( frequency of class=<class int> ) (key : value pairs)
            
            self.learned_logprob_feature_given_class: initializes | classes | x | features | array whose [r,c] entry represents the log( P( feature c | class r ) )

        """

        # some sanity checks
        
        if isinstance( binary_feature_array, csr_matrix ): 
            bool_feature_array = binary_feature_array.astype( bool ).astype( int )
            del binary_feature_array # for memory footprint
            binary_feature_array = bool_feature_array.transpose() # converts the features from column vectors to row vectors (in order to do dot product for counts (faster))
            del bool_feature_array

        if isinstance( binary_feature_array, np.ndarray ):
            bool_feature_array = binary_feature_array.astype( bool ).astype( int )
            del binary_feature_array # for memory footprint
            binary_feature_array = csr_matrix( bool_feature_array.T ) # converts the features from column vectors to row vectors (in order to do dot product for counts (faster))
            del bool_feature_array 

        if binary_class_vec.dtype != 'bool':
            binary_class_vec = binary_class_vec.astype( bool ).astype( int )

        ## if targets is a (1,#) or (#,1) array instead of a (#,) vector
        if binary_class_vec.ndim > 1:
            binary_class_vec = binary_class_vec.flatten()
     
        # compute count of (target=1) and (target=0)
        count_target1 = np.sum( binary_class_vec )
        count_target0 = len( binary_class_vec ) - count_target1

        # compute P(feature_i | target=1)
        ## dummy initializations -- no longer needed with dot product shortcut
        # prob_feat_i_cond_1 = np.array( [ 0.0 ] * binary_feature_array.shape[0] , dtype=np.float64 ) # remember that the binary_feature_array has been transposed!
        # prob_feat_i_cond_0 = np.array( [ 0.0 ] * binary_feature_array.shape[0] , dtype=np.float64 ) # remember that the binary_feature_array has been transposed!

        ## used for P( y=0 | x ), swaps Trues -> Falses
        inverted_binary_class_vec = 1.0 - ( binary_class_vec ) 
        
        if verbose:
            print("starting the fitting/learning process")

        prob_feat_i_cond_1 = binary_feature_array.dot( binary_class_vec )
        prob_feat_i_cond_0 = binary_feature_array.dot( inverted_binary_class_vec )
        '''
        for feat_i in trange( binary_feature_array.shape[0] ): # remember that the binary_feature_array has been transposed!
            
            feature_column = binary_feature_array.getrow( feat_i )

            
            # Old code for dense arrays, can lead to memory errors 
            prob_feat_i_cond_1[ feat_i ] =  np.sum( 
                                                np.logical_and( feature_column, binary_class_vec ) 
                                            )
            prob_feat_i_cond_0[ feat_i ] =  np.sum( 
                                                np.logical_and( feature_column, inverted_binary_class_vec ) 
                                            )
            Note: 
            # np.logical_and( np.array( [ True, True, False, False ] ), 
            #                 np.array( [ True, False, True, False ] ) )
            # yields True False False False
            

            #class1res = feature_column.dot( binary_class_vec )
            #assert class1res.shape == (1,)
            #prob_feat_i_cond_1[ feat_i ] = class1res[0]

            #class0res = feature_column.dot( inverted_binary_class_vec )
            #assert class0res.shape == (1,)
            #prob_feat_i_cond_0[ feat_i ] = class0res[0]
        '''
        if laplace_smoothing: 
            prob_feat_i_cond_1 = ( prob_feat_i_cond_1 + 1 ) / ( count_target1 + 2 )
            prob_feat_i_cond_0 = ( prob_feat_i_cond_0 + 1 ) / ( count_target0 + 2 )
            self.learned_logprob_feature_given_class = np.log( np.vstack( ( prob_feat_i_cond_0, prob_feat_i_cond_1 ) ) )
        
        else:
            prob_feat_i_cond_1 = ( prob_feat_i_cond_1 ) / ( count_target1 )
            prob_feat_i_cond_0 = ( prob_feat_i_cond_0 ) / ( count_target0 )
            self.learned_logprob_feature_given_class = np.log( np.vstack( ( prob_feat_i_cond_0, prob_feat_i_cond_1 ) ) )
        
        self.class_log_probs[0] = log( count_target0 / len( binary_class_vec ) )
        self.class_log_probs[1] = log( count_target1 / len( binary_class_vec ) )

    def predict( self, test_instances, verbose=True ):
        """
        Returns predicted class for test_instance using MLE/MAP parameters and class probabilities.

        Arguments:

            test_instances: | testing instances | x | features |-sized numpy array or csr_matrix (get converted to a csr_matrix either way).

            verbose: boolean indicator of verbosity.
        
        Returns:

            Binary vector of the predicted class ( 0 / 1 ) whose ith element is the class prediction for the ith testing instance.

        """
        # Note/Disclaimer: 'test_instances' and 'test_instance' could also have been named 'validation_instances' and 'validation_instance'

        # sanity checks
        if isinstance( test_instances, csr_matrix ): 
            bool_feature_array = test_instances.astype( bool ).astype( int ) # ensures a < float > 0 > -> 1 mapping
            del test_instances # for memory footprint
            test_instances = bool_feature_array

        if isinstance( test_instances, np.ndarray ):
            bool_feature_array = test_instances.astype( bool ).astype( int )
            del test_instances # for memory footprint
            test_instances = csr_matrix( bool_feature_array )

        # dummy initializations
        class_preds, ml_prob = np.array( [None]*test_instances.shape[0] ), np.array( [-inf]*test_instances.shape[0] )

        # loop over instances 
        if verbose:
            print("starting to make predictions")
            
        for test_instance_index in trange( test_instances.shape[0] ):
            test_instance_csr = test_instances[ test_instance_index, : ]
            test_instance = test_instance_csr.toarray().flatten()
        # for test_instance_index, test_instance in enumerate( test_instances ):
            # iterate over every class, keeping track of which class has the highest probability
            for row_index, row_class_logprobs in enumerate( self.learned_logprob_feature_given_class ):

                # not_row_class_probs is log( 1.0 - P( feature c | class r ) ), precomputes log( P( feature c = 0 | class r ) )
                not_row_class_probs = np.log( np.ones( row_class_logprobs.shape ) - np.exp( row_class_logprobs ) )

                probs = np.vstack( ( not_row_class_probs, row_class_logprobs ) )

                class_prob = self.class_log_probs[ row_index ] + sum( [ probs[ test_instance_feature, test_instance_index ] for test_instance_index, test_instance_feature in enumerate( test_instance ) ] )
                                
                if class_prob >= ml_prob[ test_instance_index ]:
                    class_preds[ test_instance_index ], ml_prob[ test_instance_index ] = row_index, class_prob 

        return class_preds.astype( int )


def binary_bernoulli_NB( binary_feature_array: np.ndarray, binary_class_vec: np.ndarray, laplace_smoothing=True ):
    """
    This function has been implemented in the homemade_BernoulliNB class.
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
    This function has been implemented in the homemade_BernoulliNB class.
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
    # everything below is just to test the functions/classes/methods

    # random initialization of a training dataset
    classes =  np.random.randint( 0, 2, ( 100 ) )
    features = csr_matrix( np.random.randint( 0, 2, ( 100, 10 ) ) )

    # using the functions individually, res == learned probabilities from training
    # logp0, logp1, res =  binary_bernoulli_NB( features, classes) 
    
    # sanity check with sklearn's BernoulliNB classifier
    sanity_check = BernoulliNB()
    sanity_check.fit( features, classes )

    # sanity check with homemade_BernoulliNB class
    homemade_sanity_check = homemade_BernoulliNB()
    homemade_sanity_check.fit( features, classes )
    
    # generating 1,000,000 random test datasets
    for i in range( 100 ):
        print( i )
        tests = np.random.randint( 0, 2, ( 10, 10 ) )
        
        sanity_preds = sanity_check.predict( tests )
        homemade_sanity_preds = homemade_sanity_check.predict( tests )
        
        '''my_preds = []
        for t in tests:
            my_preds.append( predict( res, [ logp0, logp1 ], t ) )
        '''
        print( homemade_sanity_preds )
        print( sanity_preds )
        #assert ( np.array_equal( np.array( my_preds ), sanity_preds ) )
        
        assert( np.array_equal( sanity_preds, homemade_sanity_preds ) )
    
