import random
import GeoIP

import numpy as np
import time
from sklearn import svm, preprocessing
import pandas as pd

# one-class SVM script

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from csv...")
df = pd.read_csv("dataset/dns_scan.csv", ) # inserire la scansione da cui prendere i dati, su github non ci sta per
# questioni di memoria
ip_addresses = df.saddr
print("...Done")

n_row = 100000  # len(ip_addresses)
n_col = 2

X_train = np.empty(shape=[n_row, n_col])
print("Generate some train data...")
for i in range(n_row):
    address = ip_addresses[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = 0
    X_train[i][0] = result
    X_train[i][1] = asn

print("...Done")

X_train_scaled = preprocessing.scale(X_train)

n_row_test = 100000
X_test = np.empty(shape=[n_row_test, n_col])
print("Generate some regular novel observations...")
for i in range(n_row):
    address = ip_addresses[n_row + i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[n_row + i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = 0
    X_test[i][0] = result
    X_test[i][1] = asn
print("...Done")

X_test_scaled = preprocessing.scale(X_test)

print("Generating some abnormal novel observation...")
X_outliers = np.empty(shape=[n_row_test, n_col])
for i in range(n_row):
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = random.randint(0, 65535)

    X_outliers[i][0] = result
    X_outliers[i][1] = asn
print("...Done")

X_outliers_scaled = preprocessing.scale(X_outliers)

print("Starting the train...")
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train_scaled)
print("...Done")

print("Starting the test...")
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)
y_pred_outliers = clf.predict(X_outliers_scaled)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
print("...Done")

print("N error train: " + str(n_error_train))
print("N error test: " + str(n_error_test))
print("N error outliers: " + str(n_error_outliers))

print("--- %s seconds ---" % (time.time() - start_time))

