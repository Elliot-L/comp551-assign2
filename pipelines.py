from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt 
from more_itertools import consecutive_groups

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