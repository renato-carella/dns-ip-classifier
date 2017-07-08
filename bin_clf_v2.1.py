#!/usr/bin/python3.5

import random
import GeoIP
import numpy as np
import sys
from sklearn import svm, linear_model, preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import time

########################################################################################################################
#
# Binary classifier
# The two features are:
# - IP address converted in binary number
# - Autonomous System
#       - address for which is not possible to retrieve AS are dropped
# RandomForest and SVC classifier are used:
# - RandomForest is tuned with the following parameters:
#   - max_features=None
#   - n_estimators=20
# - SVC standard version
#
########################################################################################################################

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from csv...")
df = pd.read_csv(sys.argv[1])
ip_addresses = df.saddr
print("...Done")

n_row_train = int(sys.argv[2])  # int(len(ip_addresses) / 2)
if n_row_train <= 0 or n_row_train > int(len(ip_addresses) / 2):
    print("The size of training set must be greater than zero and minor than IP address number of the scan in use")
    sys.exit(1)

n_row_regular_test = int(sys.argv[3])  # int(len(ip_addresses)/2)
if n_row_regular_test <= 0 or n_row_regular_test > int(len(ip_addresses)) - n_row_train:
    print("The size of n_row_regular_test must be greater than zero and minor than [(IP address number of the scan in "
          "use) - (size of training set)]")
    sys.exit(1)

n_row_outliers_test = int(sys.argv[4])
if n_row_outliers_test <= 0:
    print("The size of n_row_outliers_test must be greater than zero and minor than [(IP address number of the scan "
          "in use) - (size of training set)]")
    sys.exit(1)

classifier = sys.argv[5]
if classifier == "svm":
    clf = svm.SVC()
elif classifier == "rf":
    clf = RandomForestClassifier(max_depth=None, n_estimators=20, max_features=None)
else:
    print("Classifier not valid, options are svm of rf")
    sys.exit(1)

n_col = 2

# ------------------------------ Training Set ------------------------------


print("Generate some regular train data...")

X_train_regular_temp = np.empty(shape=[n_row_train, n_col])
regulars_counter = 0
i = 0

while regulars_counter != n_row_train:
    address = ip_addresses[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
        X_train_regular_temp[regulars_counter][0] = result
        X_train_regular_temp[regulars_counter][1] = asn
        regulars_counter += 1
    i += 1

X_train_regulars = np.empty(shape=[regulars_counter, n_col])
for i in range(regulars_counter):
    X_train_regulars[i] = X_train_regular_temp[i]

print("...Done")


print("Generating some abnormal training data...")

X_train_outliers_temp = np.empty(shape=[n_row_train, n_col])
outliers_counter = 0

while outliers_counter != n_row_train:
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
        X_train_outliers_temp[outliers_counter][0] = result
        X_train_outliers_temp[outliers_counter][1] = asn
        outliers_counter += 1

X_train_outliers = np.empty(shape=[outliers_counter, n_col])
for i in range(outliers_counter):
    X_train_outliers[i] = X_train_outliers_temp[i]

print("...Done")

X = np.empty(shape=[regulars_counter + outliers_counter, n_col])
for i in range(regulars_counter):
    X[i] = X_train_regulars[i]
for i in range(outliers_counter):
    X[regulars_counter + i] = X_train_outliers[i]

# Labels
a = np.array([0])
A = np.repeat(a, regulars_counter)
b = np.array([1])
B = np.repeat(b, outliers_counter)
Y = np.append(A, B)

# ------------------------------ Test Set ------------------------------

H_test_regular_temp = np.empty(shape=[n_row_regular_test, n_col])
regulars_counter = 0
i = 0

print("Generate some regular test data...")
while regulars_counter != n_row_regular_test:
    address = ip_addresses[n_row_train + i].split(".")
    # print(ip_addresses[n_row_train + i])
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[n_row_train + i])
    if gio is not None:
        # print("Ok")
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
        # print("asn: " + str(asn))
        H_test_regular_temp[regulars_counter][0] = result
        H_test_regular_temp[regulars_counter][1] = asn
        regulars_counter += 1
        # print("Regulars counter" + str(regulars_counter))
    i += 1

H_test_regulars = np.empty(shape=[regulars_counter, n_col])
for i in range(regulars_counter):
    H_test_regulars[i] = H_test_regular_temp[i]

print("...Done")


print("Generating some abnormal test data...")

H_test_outliers_temp = np.empty(shape=[n_row_outliers_test, n_col])
outliers_counter = 0

while outliers_counter != n_row_outliers_test:
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
        H_test_outliers_temp[outliers_counter][0] = result
        H_test_outliers_temp[outliers_counter][1] = asn
        outliers_counter += 1

H_test_outliers = np.empty(shape=[outliers_counter, n_col])
for i in range(outliers_counter):
    H_test_outliers[i] = H_test_outliers_temp[i]

print("...Done")
# print("H_test_regulars length: " + str(len(H_test_regulars)))
# print("H_test_outliers length: " + str(len(H_test_outliers)))

H = np.empty(shape=[regulars_counter + outliers_counter, n_col])
for i in range(regulars_counter):
    H[i] = H_test_regulars[i]
for i in range(outliers_counter):
    H[regulars_counter + i] = H_test_outliers[i]

# Labels
a = np.array([0])
A = np.repeat(a, regulars_counter)
b = np.array([1])
B = np.repeat(b, outliers_counter)
I = np.append(A, B)

# ----------------------------------------------------------------------

# SKLEARN #



print("Training...")
clf.fit(X, Y)
print("Training finished")
print("Starting the test...")
print("Score on test set: {0}\n".format(clf.score(H, I)))

# print("Measuring other metrics...")
# TP = 0
# FP = 0
# TN = 0
# FN = 0
#
# result = clf.predict(H)
#
# for i in range(n_row_test*2):
#     if I[i] == result[i] == 0:
#         TP += 1
#     if result[i] == 0 and result[i] != I[i]:
#         FP += 1
#     if I[i] == result[i] == 1:
#         TN += 1
#     if result[i] == 1 and result[i] != I[i]:
#         FN += 1
# #
# print("True positive: " + str(TP))
# print("False positive: " + str(FP))
# print("True negative = " + str(TN))
# print("False negative = " + str(FN))
#
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# print("Precision: {0}".format(precision))
# print("Recall: {0}".format(recall))
# print("F-Measure: {0}".format(2*((precision*recall)/(precision+recall))))

print("--- %s seconds ---" % (time.time() - start_time))
