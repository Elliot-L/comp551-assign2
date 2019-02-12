from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
import json
from pipelines import k_fold

features = list()
labels = list()
with open("train_tfidf.txt") as data_file:
    csv_reader = csv.reader(data_file, delimiter=',')
    for row in csv_reader:
        features.append(np.array(json.loads(row[0])))
        labels.append(np.array(json.loads(row[1])))
X = np.array(features)
y = np.array(labels)
print('done')

splits = k_fold(X, y, k=2)
num_sets = len(splits[0])
accuracies = list()
for i in range(num_sets):
    train_x = splits[0][i]
    valid_x = splits[1][i]
    train_y = splits[2][i]
    valid_y = splits[3][i]
    model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(train_x, train_y)

    # validate
    predictions = model.predict(valid_x)

    # compute accuracy
    num_correct = 0.0
    for i in range(len(predictions)):
        if predictions[i] == valid_y[i]:
            num_correct += 1.0
    set_accuracy = num_correct / float(len(predictions))
    print("Set accuracy: " + str(set_accuracy))
    accuracies.append(set_accuracy)

print("Total accuracy: " + str(np.average(accuracies)))


