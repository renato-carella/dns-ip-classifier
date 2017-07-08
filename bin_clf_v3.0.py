#!/usr/bin/python3.5

import numpy as np
import sys
from sklearn import svm, preprocessing
from sklearn.model_selection import KFold
import GeoIP
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import time

from sklearn.tree import DecisionTreeClassifier

########################################################################################################################
#
# Binary classifier with KFold
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

n_row_regular = int(sys.argv[2])  # int(len(ip_addresses) / 2)
if n_row_regular <= 0 or n_row_regular > int(len(not_dns_ip)):
    print("The size of regular set must be greater than zero and minor than IP address number of the DNS scan in use")
    sys.exit(1)

n_row_abnormal = int(sys.argv[3])  # int(len(ip_addresses)/2)
if n_row_abnormal <= 0 or n_row_abnormal > int(len(dns_ip)):
    print("The size of abnormal set must be greater than zero and minor than IP address number of the non-DNS scan in "
          "use")
    sys.exit(1)

classifier = sys.argv[4]
if classifier == "svm":
    clf = svm.SVC()
elif classifier == "rf":
    clf = RandomForestClassifier(max_depth=None, n_estimators=20, max_features=None)
else:
    print("Classifier not valid, options are svm of rf")
    sys.exit(1)

n_col = 2

# ------------------------------ Data set ------------------------------

X = np.empty(shape=[n_row_regular + n_row_abnormal, n_col])

print("Generate some regular train data...")
for i in range(n_row_regular):
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


print("Generate some abnormal train data...")
for i in range(n_row_abnormal):
    address = not_dns_ip[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(not_dns_ip[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = 0

    X[n_row_regular + i][0] = result
    X[n_row_regular + i][1] = asn
print("...Done")


# Labels
a = np.array([0])
A = np.repeat(a, n_row_regular)
b = np.array([1])
B = np.repeat(b, n_row_abnormal)
Y = np.append(A, B)

# ----------------------------------------------------------------------

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
