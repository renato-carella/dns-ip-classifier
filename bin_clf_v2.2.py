#!/usr/bin/python3.5

import GeoIP
import sys
import numpy as np
from sklearn import svm, linear_model, preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import time

########################################################################################################################
#
# Binary classifier without KFold
# The two features are:
# - IP address converted in binary number
# - Autonomous System
#       - equal to 0 for those address for which is not possible to retrieve it
# RandomForest and SVC classifier are used:
# - RandomForest is tuned with the following parameters:
#   - max_features=None
#   - n_estimators=20
# - SVC standard version
# non-DNS addresses are taken from Telecom dataset
#
########################################################################################################################

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from csv...")
df = pd.read_csv(sys.argv[1], )
dns_ip = df.saddr
dns_ip = shuffle(dns_ip)
print("...Done")

print("Reading data from non-dns csv...")
rf = pd.read_csv("dataset/not-dns.csv", )  # inserire la scansione da cui prendere i dati, su github non ci sta per
not_dns_ip = rf.saddr
not_dns_ip = shuffle(not_dns_ip)
print("...Done")

n_row_train = int(sys.argv[2])  # int(len(ip_addresses) / 2)
if n_row_train <= 0 or n_row_train > int(len(not_dns_ip) / 2):
    print("The size of training set must be greater than zero and minor than IP address number of the scan in use")
    sys.exit(1)

n_row_regular_test = int(sys.argv[3])  # int(len(ip_addresses)/2)
if n_row_regular_test <= 0 or n_row_regular_test > int(len(dns_ip)) - n_row_train:
    print("The size of n_row_regular_test must be greater than zero and minor than [(IP address number of the scan in "
          "use) - (size of training set)]")
    sys.exit(1)

n_row_outliers_test = int(sys.argv[4])
if n_row_outliers_test <= 0 or n_row_outliers_test > int(len(not_dns_ip)) - n_row_train:
    print("The size of n_row_regular_test must be greater than zero and minor than [(IP address number of the scan in "
          "use) - (size of training set)]")
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

X = np.empty(shape=[n_row_train * 2, n_col])

print("Generate some regular train data...")
for i in range(n_row_train):
    address = dns_ip[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(dns_ip[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = 0

    X[i][0] = result
    X[i][1] = asn
print("...Done")


print("Generating some abnormal training data...")
for i in range(n_row_train):
    address = not_dns_ip[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(not_dns_ip[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = 0
    X[n_row_train + i][0] = result
    X[n_row_train + i][1] = asn
print("...Done")

# Labels
a = np.array([0])
A = np.repeat(a, n_row_train)
b = np.array([1])
B = np.repeat(b, n_row_train)
Y = np.append(A, B)

# ------------------------------ Test Set ------------------------------

H = np.empty(shape=[n_row_regular_test + n_row_outliers_test, n_col])

print("Generate some regular test data...")
for i in range(n_row_regular_test):
    address = dns_ip[n_row_train + i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(dns_ip[n_row_train + i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = 0
    H[i][0] = result
    H[i][1] = asn
print("...Done")


print("Generating some abnormal test data...")
for i in range(n_row_outliers_test):
    address = not_dns_ip[n_row_train + i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(dns_ip[n_row_train + i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = 0
    H[i][0] = result
    H[i][1] = asn
print("...Done")


# Labels
a = np.array([0])
A = np.repeat(a, n_row_regular_test)
b = np.array([1])
B = np.repeat(b, n_row_outliers_test)
I = np.append(A, B)

# ----------------------------------------------------------------------

# SKLEARN #

if classifier == "svm":
    print("Support Vector Machine")
    X = preprocessing.scale(X)
    H = preprocessing.scale(H)
else:
    print("Random Forest Classifier")

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
