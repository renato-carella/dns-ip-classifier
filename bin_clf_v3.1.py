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
# Binary classifier with KFold
# The two features are:
# - IP address converted in binary number
# - Autonomous System
#       - address for which is not possible to retrieve it are dropped
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

print("Reading data from dns csv...")
df = pd.read_csv(sys.argv[1], )
dns_ip = df.saddr
dns_ip = shuffle(dns_ip)
print("...Done")

print("Reading data from non-dns csv...")
rf = pd.read_csv("dataset/not-dns.csv", )  # inserire la scansione da cui prendere i dati, su github non ci sta per
not_dns_ip = rf.saddr
not_dns_ip = shuffle(not_dns_ip)
print("...Done")

classifier = sys.argv[2]
if classifier == "svm":
    clf = svm.SVC()
elif classifier == "rf":
    clf = RandomForestClassifier(max_depth=None, n_estimators=20, max_features=None)
else:
    print("Classifier not valid, options are svm of rf")
    sys.exit(1)

n_col = 2
n_row = len(not_dns_ip)

# ------------------------------ Training Set ------------------------------


print("Generating some abnormal training data...")

X_outliers_temp = np.empty(shape=[n_row, n_col])
outliers_counter = 0

for i in range(n_row):
    address = not_dns_ip[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(not_dns_ip[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
        X_outliers_temp[outliers_counter][0] = result
        X_outliers_temp[outliers_counter][1] = asn
        outliers_counter += 1

# print("Outliers counter: " + str(outliers_counter))
print("...Done")

X = np.empty(shape=[outliers_counter * 2, n_col])
for i in range(outliers_counter):
    X[i] = X_outliers_temp[i]

print("Generate some regular train data...")
regulars_counter = 0
i = 0

while regulars_counter != outliers_counter:
    address = dns_ip[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(dns_ip[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
        X[outliers_counter + regulars_counter][0] = result
        X[outliers_counter + regulars_counter][1] = asn
        regulars_counter += 1
    i += 1
print("...Done")


# Labels
a = np.array([1])
A = np.repeat(a, outliers_counter)
b = np.array([0])
B = np.repeat(b, regulars_counter)
Y = np.append(A, B)

# ------------------------------ Test Set ------------------------------

# Initialize the classifier

if classifier == "svm":
    print("Support Vector Machine")
    X = preprocessing.scale(X)
    H = preprocessing.scale(H)
else:
    print("Random Forest Classifier")

kf = KFold(n_splits=4, shuffle=True)
for train, test in kf.split(X):
    X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
    print("Training...")
    clf.fit(X_train, Y_train)
    print("Training finished")
    print("Starting the test...")
    print("Score on test set: {0}\n".format(clf.score(X_test, Y_test)))


print("--- %s seconds ---" % (time.time() - start_time))
