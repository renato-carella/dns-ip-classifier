import csv

import GeoIP
import random

import pandas as pd
import numpy as np
import time
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Semplice classificatore che come features gli ottetti e non il numero biario, con AS

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from csv...")
df = pd.read_csv("dataset/dns_scan.csv", )
ip_addresses = df.saddr
print("...Done")

n_col = 5

# ------------------------------ Training Set ------------------------------

n_row_train = 100000  # int(len(ip_addresses) / 2)

X = np.empty(shape=[n_row_train * 2, n_col])
# print(X)

print("Generate some regular train data...")
for i in range(n_row_train):
    address = ip_addresses[i].split(".")
    result = np.array([float(x)/255 for x in address])
    gio = gi_asn.org_by_addr(ip_addresses[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = 0

    for j in range(4):
        X[i][j] = result[j]
    X[i][4] = asn
print("...Done")

print("Generating some abnormal training data...")
for i in range(n_row_train):
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = np.array([float(x)/255 for x in address])
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = random.randint(0, 65535)
    for j in range(4):
        X[i + n_row_train][j] = result[j]
    X[i + n_row_train][4] = asn
print("...Done")

X_scaled = preprocessing.scale(X)

# Labels
a = np.array([0])
A = np.repeat(a, n_row_train)
b = np.array([1])
B = np.repeat(b, n_row_train)
Y = np.append(A, B)


# ------------------------------ Test Set ------------------------------

n_row_test = 100000  # int(len(ip_addresses) / 2)

H = np.empty(shape=[n_row_test * 2, n_col])

print("Generate some regular test data...")
for i in range(n_row_test):
    address = ip_addresses[n_row_train + i].split(".")
    result = np.array([float(x)/255 for x in address])
    gio = gi_asn.org_by_addr(ip_addresses[n_row_train + i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = 0

    for j in range(4):
        H[i][j] = result[j]
    H[i][4] = asn
print("...Done")

print("Generating some abnormal test data...")
for i in range(n_row_test):
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = np.array([float(x)/255 for x in address])
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = random.randint(0, 65535)

    for j in range(4):
        H[i + n_row_test][j] = result[j]
    H[i + n_row_test][4] = asn
print("...Done")

H_scaled = preprocessing.scale(H)
# print(H_scaled.mean(axis=0))
# print(H_scaled.std(axis=0))

# Labels
a = np.array([0])
A = np.repeat(a, n_row_test)
b = np.array([1])
B = np.repeat(b, n_row_test)
I = np.append(A, B)
# print (I)
# print (I.shape)

# SKLEARN #

# Initialize the classifier
# clf = linear_model.SGDClassifier()
rfc = RandomForestClassifier(max_depth=None, n_estimators=10, max_features=None)

# print("LINEAR MODEL")
# print("Training...")
# clf.fit(X, Y)
# print("Training finished")
# print("Score on test set: {0}\n".format(clf.score(H, I)))

print("RANDOM FOREST CLASSIFIER")
print("Training...")
rfc.fit(X, Y)
print("Training finished")
print("Score on test set: {0}\n".format(rfc.score(H, I)))

print("--- %s seconds ---" % (time.time() - start_time))
