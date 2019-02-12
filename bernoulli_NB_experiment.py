import numpy as np 
from math import log, e, inf
import csv, json
from sklearn.naive_bayes import BernoulliNB
from bernoulli_NB import homemade_BernoulliNB
from pipelines import k_fold

def loader():
    features, labels = [],[]
    with open("train_tfidf.txt", 'r', encoding="utf8", newline='\n') as data_file:
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

def run_experiment( X_float, y_float ):

    X = ( X_float > 0 ).astype( int )
    y = ( y_float > 0 ).astype( int )
    
    splits = k_fold(X, y, k=5) 
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
        predictions = bernie.predict( valid_x )

        # sanity checks
        sanity_check = BernoulliNB()
        sanity_check.fit( train_x, train_y )
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
    X, y = loader()
    run_experiment( X, y )

if __name__ == '__main__':
    main()