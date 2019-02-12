from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
import numpy as np


def k_fold(X,y):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    return X_train_list,X_test_list,y_train_list,y_test_list


def strat_k_fold(X,y):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    kf = StratifiedKFold(n_splits=2)
    for train_index, test_index in kf.split(X):
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