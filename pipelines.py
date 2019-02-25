from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt 
from more_itertools import consecutive_groups
from scipy.sparse import csr_matrix, hstack

def kfold_plotter( training_indices, testing_indices ):

    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=1 )
    bar_width=0.4
    bar_positions = np.arange( len( training_indices ) )
    for i, ( cv_train, cv_test ) in enumerate( zip( training_indices, testing_indices ) ):
        cv_train += 1
        cv_test += 1
        print( cv_train )
        X_train_groups = [ list( group ) for group in consecutive_groups( cv_train ) ]
        X_test_groups = [ list( group ) for group in consecutive_groups( cv_test ) ]
        print( X_train_groups )
        print( X_test_groups )
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

def add_n_most_common_words_features( features_array: np.ndarray, new_feature_array: np.ndarray, sparse=False ):
    """
    S.C.
    Compiles a word frequency array for the instances in features_array and the n most common words 
    and appends it to the features_array provided.

    Arguments:

        features_array: the | instances | x | features | numpy array to which the word frequency array will be appended.

        new_feature_array: the | instances | x | new features | numpy array to append to the features_array.

        sparse: boolean indicator of whether the input and output arrays are sparse matrices.
    
    Returns:

        [ [features_array] [topn_word_frequency_array] ]
    """
        
    assert features_array.shape[0] == new_feature_array.shape[0]

    if sparse:
        return hstack( [features_array, new_feature_array] )
    else:
        return np.hstack( ( features_array, new_feature_array ) )
    
def add_transformed_feature( features_array, feature_column_index, transform, sparse=False, *args, **kwargs ):
    """
    S.C.
    Function to apply "transform" to the feature_column_index-th feature column of features_array and append its
    resulting vector to the original features_array.

    Example of use: 
    
    1) To append a feature which is the log of the 2nd column in features_array:
        transformation_funct = lambda vector:np.log( vector )
        updated_feature_array = add_transformed_feature( current_feature_array, 1, transformation_funct ) 

    2) To append a feature which is a product of the 2nd and 4th feature vectors ( additional transformations can be chained in here ):
        transformation_funct = lambda column_index_1, column_index_2 : np.multiply( column_index_1, column_index_2 )
        updated_feature_array = add_transformed_feature( current_feature_array, 2, transformation_funct, column_index_2=current_feature_array[ :, 3] )

    Arguments:

        features_array: np.ndarray of the original feature array. 

        feature_column_index: integer index of the feature on which "transform" will be applied. 

        transform: a lambda function version of the transform to apply ( needs to be applicable to numpy type objects ). 

        sparse: boolean indicator of whether the input and output arrays are sparse matrices. 

    Returns:

        [ [ features_array ] [ transformed feature ] ] representation of updated features_array.

    """
    # assert feature_column_index < features_array.shape[1]
    
    if not sparse: 
        transformed_feature = transform( features_array[ :, feature_column_index ], *args, **kwargs )
        transformed_feature = np.reshape( transformed_feature, ( len( transformed_feature ) , 1 ) )
        return np.hstack( ( features_array, transformed_feature ) )
    else:
        dense_features_array = features_array.toarray()
        transformed_feature = transform( dense_features_array[ :, feature_column_index ], *args, **kwargs )
        transformed_feature = np.reshape( transformed_feature, ( len( transformed_feature ) , 1 ) )
        transformed_features_array = np.hstack( ( features_array, transformed_feature ) )
        return csr_matrix( transformed_features_array )

def leave_one_out(X,y):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train)
        print(y_train)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    return X_train_list, X_test_list, y_train_list, y_test_list

def main():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    k_fold(X,y)
    strat_k_fold(X,y)
    leave_one_out(X,y)


if __name__ == "__main__":
    main()