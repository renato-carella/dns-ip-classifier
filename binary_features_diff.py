import random
import GeoIP

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# Questo script allena il classificatore con una scansione più vecchia di un anno rispetto alla scansione utilizzata
# come test set

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from last csv...")
df_last = pd.read_csv("dataset/dns_scan.csv", )  # inserire la scansione da cui prendere i dati, su github
# non ci sta per questioni di memoria
ip_addresses_last = df_last.saddr
print("...Done")

print("Reading data from an old csv...")
df_old = pd.read_csv("dataset/dns_scan_old.csv", )  # inserire la scansione precedente da cui prendere i dati, su github
# non ci sta per questioni di memoria
ip_addresses_old = df_old.saddr
print("...Done")

n_col = 2

# ------------------------------ Training Set ------------------------------

n_row_train = 100000  # len(ip_addresses_old)


X = np.empty(shape=[n_row_train * 2, n_col])

print("Generate some regular train data...")
for i in range(n_row_train):
    address = ip_addresses_old[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses_old[i])
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
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
    # gir = gi.record_by_addr(ip)
    # if gir is not None:
    #     latitude = gir['latitude'] + 60
    #     longitude = gir['longitude'] + 60
    # else:
    #     latitude = 0
    #     longitude = 0
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

# # ------------------------------ Test Set ------------------------------

n_row_test = 100000  # len(ip_addresses_last)
H = np.empty(shape=[n_row_test * 2, n_col])

print("Generate some regular test data...")
for i in range(int(n_row_test)):
    address = ip_addresses_last[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses_last[i])
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

print("--- %s seconds ---" % (time.time() - start_time))
