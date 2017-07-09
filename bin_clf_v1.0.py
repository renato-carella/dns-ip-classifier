#!/usr/bin/python3.5

import time
import sys
import random
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier


########################################################################################################################
#
#
# Binary classifier without KFold
# The only feature is the IP address converted in binary number
# RandomForest and SVC classifier are used in the standard version
#
#
########################################################################################################################


def read_data(scan_file):
    print("Reading data from csv...")
    df = pd.read_csv(scan_file, )
    ip_addresses = df.saddr
    ip_addresses = shuffle(ip_addresses)
    print("...Done")
    return ip_addresses


def check_params(n_row_training_set, n_row_regulars_test_set, n_row_outliers_test_set, scan_length):
    if n_row_training_set <= 0 or n_row_train > int(scan_length / 2):
        print("The size of training set must be greater than zero and minor than IP address number of the scan in use")
        sys.exit(1)

    if n_row_regulars_test_set <= 0 or n_row_regulars_test_set > scan_length - n_row_train:
        print("The size of n_row_regular_test must be greater than zero and minor than [(IP address number of the scan "
              "in use) - (size of training set)]")
        sys.exit(1)

    if n_row_outliers_test_set <= 0:
        print(
            "The size of n_row_outliers_test must be greater than zero and minor than [(IP address number of the scan "
            "in use) - (size of training set)]")
        sys.exit(1)


def set_classifier(clf_string):
    if clf_string == "svm":
        classifier = svm.SVC()
    elif clf_string == "rf":
        classifier = RandomForestClassifier(max_depth=None, n_estimators=10, max_features='auto')
    else:
        print("Classifier not valid, options are svm of rf")
        sys.exit(1)
    return classifier


def create_training_set(n_row, ip_addresses):
    training_set = np.empty(shape=[n_row * 2, n_col])

    print("Generate some regular train data...")
    for i in range(n_row_train):
        result = convert_ip(ip_addresses[i])
        training_set[i][0] = result
    print("...Done")

    print("Generating some abnormal training data...")
    for i in range(n_row_train):
        ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
        result = convert_ip(ip)
        training_set[n_row + i][0] = result
    print("...Done")
    return training_set


def convert_ip(ip_address):
    address = ip_address.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    return result


def assign_labels(regulars, outliers):
    a = np.array([0])
    A = np.repeat(a, regulars)
    b = np.array([1])
    B = np.repeat(b, outliers)
    labels_set = np.append(A, B)
    return labels_set


def create_test_set(n_regulars, n_outliers, n_train, scan):
    test_set = np.empty(shape=[n_regulars + n_outliers, n_col])

    print("Generate some regular test data...")
    for i in range(n_regulars):
        result = convert_ip(scan[n_train + i])
        test_set[i][0] = result
    print("...Done")

    print("Generating some abnormal test data...")
    for i in range(n_outliers):
        ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
        result = convert_ip(ip)
        test_set[n_regulars + i][0] = result
    print("...Done")
    return test_set


def scale_data(data_set):
    return preprocessing.scale(data_set)


def train_classifier(classifier, training_set, labels):
    print("Training...")
    classifier.fit(training_set, labels)
    print("Training finished")
    return clf


def test_classifier(classifier, test_set, labels, measure='accuracy'):
    print("Starting the test...")

    if measure == 'accuracy':
        return classifier.score(test_set, labels)

    tp = fp = tn = fn = 0

    result = classifier.predict(test_set)
    for i in range(len(labels)):
        if labels[i] == result[i] == 0:
            tp += 1
        if result[i] == 0 and result[i] != labels[i]:
            fp += 1
        if I[i] == result[i] == 1:
            tn += 1
        if result[i] == 1 and result[i] != I[i]:
            fn += 1

    # print("True positive: " + str(tp))
    # print("False positive: " + str(fp))
    # print("True negative = " + str(tn))
    # print("False negative = " + str(fn))

    if measure == 'precision':
        return tp / (tp + fp)
    if measure == 'recall':
        return tp / (tp + fn)
    if measure == 'f-measure':
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * ((precision * recall) / (precision + recall))


# ------------------------------ Main ------------------------------

start_time = time.time()

n_col = 1

scan_file_name = sys.argv[1]
scan_ip_addresses = read_data(scan_file_name)

n_row_train = int(sys.argv[2])
n_row_regulars_test = int(sys.argv[3])
n_row_outliers_test = int(sys.argv[4])
check_params(n_row_train, n_row_regulars_test, n_row_outliers_test, len(scan_ip_addresses))

clf_name = sys.argv[5]
clf = set_classifier(clf_name)

X = create_training_set(n_row_train, scan_ip_addresses)
Y = assign_labels(n_row_train, n_row_train)

H = create_test_set(n_row_regulars_test, n_row_outliers_test, n_row_train, scan_ip_addresses)
I = assign_labels(n_row_regulars_test, n_row_outliers_test)

if clf_name == "svm":
    print("Support Vector Machine")
    X = scale_data(X)
    H = scale_data(H)
else:
    print("Random Forest Classifier")

clf = train_classifier(clf, X, Y)

if len(sys.argv) == 6:
    score = test_classifier(clf, H, I)
    print("Score on test set: {0}\n".format(score))

if len(sys.argv) == 7:
    metric = sys.argv[6]
    score = test_classifier(clf, H, I, metric)
    if metric == 'precision':
        print("Precision: {0}".format(score))
    if metric == 'recall':
        print("Recall: {0}".format(score))
    if metric == 'f-measure':
        print("F-Measure: {0}\n".format(score))

print("--- %s seconds ---" % (time.time() - start_time))
