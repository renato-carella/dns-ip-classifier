#!/usr/bin/python3.5

import numpy as np
import sys
from sklearn import svm, preprocessing
from sklearn.model_selection import KFold
import GeoIP
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.utils import shuffle

########################################################################################################################
#
# Decimal classifier with KFold
# The five features are:
# - the four decimal number values of IP octets
# - Autonomous System
#       - address for which is not possible to retrieve AS are dropped
# RandomForest and SVC classifier are used:
# - RandomForest is tuned with the following parameters:
#   - max_features=None
#   - n_estimators=20
# - SVC standard version
# non-DNS addresses are taken from Telecom dataset
#
########################################################################################################################


def read_data(scan_file):
    print("Reading data from csv...")
    df = pd.read_csv(scan_file, )
    ip_addresses = df.saddr
    ip_addresses = shuffle(ip_addresses)
    print("...Done")
    return ip_addresses


def check_params(n_regulars, n_outliers, regulars_scan_length, outliers_scan_length):
    if n_regulars <= 0 or n_regulars > regulars_scan_length:
        print("The size of n_regulars must be greater than zero and minor than IP address regulars scan length")
        sys.exit(1)

    if n_outliers <= 0 or n_outliers > outliers_scan_length:
        print("The size of n_outliers must be greater than zero and minor than IP address outlier scan length")
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


def create_data_set(n_regulars, n_outliers, regulars_ip, outliers_ip):
    print("Generate some regular train data...")
    regulars_temp = np.empty(shape=[n_regulars, n_col])
    regulars_counter = 0

    for i in range(n_regulars):
        result = convert_ip(regulars_ip[i])
        asn = retrieve_as(regulars_ip[i])
        if asn != 0:
            for j in range(4):
                regulars_temp[regulars_counter][j] = result[j]
            regulars_temp[regulars_counter][4] = asn
            regulars_counter += 1
    print("...Done")

    print("Generating some abnormal training data...")
    outliers_temp = np.empty(shape=[n_outliers, n_col])
    outliers_counter = 0

    for i in range(n_outliers):
        result = convert_ip(outliers_ip[i])
        asn = retrieve_as(outliers_ip[i])
        if asn != 0:
            for j in range(4):
                outliers_temp[outliers_counter][j] = result[j]
            outliers_temp[outliers_counter][4] = asn
            outliers_counter += 1
    print("...Done")

    data_set = np.empty(shape=[regulars_counter + outliers_counter, n_col])
    for i in range(regulars_counter):
        data_set[i] = regulars_temp[i]
    for i in range(outliers_counter):
        data_set[regulars_counter + i] = outliers_temp[i]

    return data_set


def convert_ip(ip_address):
    address = ip_address.split(".")
    result = np.array([float(x) / 255 for x in address])
    return result


def retrieve_as(ip_address):
    gio = gi_asn.org_by_addr(ip_address)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = 0
    return asn


def assign_labels(regulars, outliers):
    a = np.array([0])
    A = np.repeat(a, regulars)
    b = np.array([1])
    B = np.repeat(b, outliers)
    labels_set = np.append(A, B)
    return labels_set


def scale_data(data_set):
    return preprocessing.scale(data_set)


def k_fold(splits, data_set, labels):
    kf = KFold(n_splits=splits, shuffle=True)
    for train, test in kf.split(X):
        X_train, X_test, Y_train, Y_test = data_set[train], data_set[test], labels[train], labels[test]
        print("Training...")
        clf.fit(X_train, Y_train)
        print("Training finished")
        print("Starting the test...")
        print("Score on test set: {0}\n".format(clf.score(X_test, Y_test)))


# ------------------------------ Main ------------------------------

start_time = time.time()

# gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

n_col = 5

dns_scan = sys.argv[1]
dns_ip = read_data(dns_scan)

not_dns_scan = sys.argv[2]
not_dns_ip = read_data(not_dns_scan)

n_row_regulars = int(sys.argv[3])
n_row_outliers = int(sys.argv[4])
check_params(n_row_regulars, n_row_outliers, len(dns_ip), len(not_dns_ip))

clf_name = sys.argv[5]
clf = set_classifier(clf_name)

X = create_data_set(n_row_regulars, n_row_outliers, dns_ip, not_dns_ip)
Y = assign_labels(n_row_regulars, n_row_outliers)

if clf_name == "svm":
    print("Support Vector Machine")
    X = scale_data(X)
else:
    print("Random Forest Classifier")

k_fold(4, X, Y)


print("--- %s seconds ---" % (time.time() - start_time))
