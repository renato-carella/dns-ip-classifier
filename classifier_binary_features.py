import random
import GeoIP

import numpy as np
from sklearn import svm, linear_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# Semplice classificatore binario senza KFold, con AS

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from csv...")
df = pd.read_csv("dataset/dns_scan.csv", ) # inserire la scansione da cui prendere i dati, su github non ci sta per
# questioni di memoria
ip_addresses = df.saddr
print("...Done")

n_col = 2

# ------------------------------ Training Set ------------------------------

n_row_train = 100000  # int(len(ip_addresses) / 2)


X = np.empty(shape=[n_row_train * 2, n_col])

print("Generate some regular train data...")
for i in range(n_row_train):
    address = ip_addresses[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[i])
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
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = random.randint(0, 65535)

    X[i + n_row_train][0] = result
    X[i + n_row_train][1] = asn
print("...Done")

# Labels
a = np.array([0])
A = np.repeat(a, n_row_train)
b = np.array([1])
B = np.repeat(b, n_row_train)
Y = np.append(A, B)

# ------------------------------ Test Set ------------------------------

n_row_test = 100000  # int(len(ip_addresses)/2)
H = np.empty(shape=[n_row_test * 2, n_col])

print("Generate some regular test data...")
for i in range(int(n_row_test)):
    address = ip_addresses[n_row_train + i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[n_row_train + i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = 0

    H[i][0] = result
    H[i][1] = asn
print("...Done")

print("Generating some abnormal test data...")
for i in range(n_row_test):
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = random.randint(0, 65535)

    H[i + n_row_test][0] = result
    H[i + n_row_test][1] = asn
print("...Done")

# Labels
a = np.array([0])
A = np.repeat(a, int(n_row_test))
b = np.array([1])
B = np.repeat(b, n_row_test)
I = np.append(A, B)
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
print("Starting the test...")
print("Score on test set: {0}\n".format(rfc.score(H, I)))

print("Measuring other metrics...")
TP = 0
FP = 0
TN = 0
FN = 0

result = rfc.predict(H)

for i in range(n_row_test*2):
    if I[i] == result[i] == 0:
        TP += 1
    if result[i] == 0 and result[i] != I[i]:
        FP += 1
    if I[i] == result[i] == 1:
        TN += 1
    if result[i] == 1 and result[i] != I[i]:
        FN += 1
#
print("True positive: " + str(TP))
print("False positive: " + str(FP))
print("True negative = " + str(TN))
print("False negative = " + str(FN))

precision = TP/(TP+FP)
recall = TP/(TP+FN)
print("Precision: {0}".format(precision))
print("Recall: {0}".format(recall))
print("F-Measure: {0}".format(2*((precision*recall)/(precision+recall))))

print("--- %s seconds ---" % (time.time() - start_time))
