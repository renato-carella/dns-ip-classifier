import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import random
import GeoIP
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

start_time = time.time()

gi = GeoIP.open("/usr/share/GeoIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
gi_asn = GeoIP.open("/usr/share/GeoIP/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)

print("Reading data from csv...")
df = pd.read_csv("dataset/dns_scan.csv", )  # inserire la scansione da cui prendere i dati, su github non ci sta per
# questioni di memoria
ip_addresses = df.saddr
print("...Done")

n_col = 2
n_row = 100000  # number of elements for the dataset

# ------------------------------ Training Set ------------------------------

X = np.empty(shape=[n_row * 2, n_col])

print("Generate some regular train data...")
for i in range(n_row):
    address = ip_addresses[i].split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip_addresses[i])
    if gio is not None:
        # take only the int number of AS
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))
    else:
        asn = 0
    X[i][0] = result
    X[i][1] = asn
print("...Done")

print("Generating some abnormal training data...")
for i in range(n_row):
    ip = '.'.join(map(str, ([random.randint(0, 255) for _ in range(4)])))
    address = ip.split(".")
    result = ''.join(map(str, ["{0:08b}".format(int(x)) for x in address]))
    gio = gi_asn.org_by_addr(ip)
    if gio is not None:
        as_split = gio.split(' ')
        asn = int(as_split[0].replace('AS', ""))  # /65535
    else:
        asn = random.randint(0, 65535)
    X[i + n_row][0] = result
    X[i + n_row][1] = asn
print("...Done")

# Labels
a = np.array([0])
A = np.repeat(a, n_row)
b = np.array([1])
B = np.repeat(b, n_row)
Y = np.append(A, B)

# Initialize the classifier

# clf = linear_model.SGDClassifier()
# clf = svm.NuSVC()
clf = RandomForestClassifier(max_depth=None, n_estimators=10, max_features=None)

# Compute the accuracy

kf = KFold(n_splits=4, shuffle=True)
for train, test in kf.split(X):
    # print("%s %s" % (train, test))
    X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
    print("Training...")
    clf.fit(X_train, Y_train)
    print("Training finished")
    print("Starting the test...")
    print("Score on test set: {0}\n".format(clf.score(X_test, Y_test)))


print("--- %s seconds ---" % (time.time() - start_time))
