from os import makedirs
import numpy as np
import svm

np.random.seed(33)

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "./bank-note/"

train_x = []
train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(-1 if terms_flt[-1] == 0 else 1)

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(-1 if terms_flt[-1] == 0 else 1)

test_x = np.array(test_x)
test_y = np.array(test_y)

print("==== Primal SVM, a) ====")
Cs = [100/873, 500/873, 700/873]
for C in Cs:
    print(f"C = {C}")
    lnot, a = 1, 1
    lr_schedule = lambda e : lnot / (1 + (lnot/a)*e)
    psvm = svm.PrimalSVM(train_x, train_y, lr_schedule=lr_schedule, C=C, epochs=100)
    print(f"learned weights: {psvm.weights[1:]}")
    print(f"learned bias: {psvm.weights[0]}")
    print(f"training accuracy: {np.mean(train_y == psvm.predict(train_x))}")
    print(f"testing accuracy: {np.mean(test_y == psvm.predict(test_x))}")

print("==== Primal SVM, b) ====")
Cs = [100/873, 500/873, 700/873]
for C in Cs:
    print(f"C = {C}")
    lnot = 1
    lr_schedule = lambda e : lnot / (1 + e)
    psvm = svm.PrimalSVM(train_x, train_y, lr_schedule=lr_schedule, C=C, epochs=100)
    print(f"learned weights: {psvm.weights[1:]}")
    print(f"learned bias: {psvm.weights[0]}")
    print(f"training accuracy: {np.mean(train_y == psvm.predict(train_x))}")
    print(f"testing accuracy: {np.mean(test_y == psvm.predict(test_x))}")

print("==== Dual SVM, a) ====")
Cs = [100/873, 500/873, 700/873]
for C in Cs:
    print(f"C = {C}")
    dsvm = svm.DualSVM(train_x, train_y, C=C)
    print(f"learned weights: {dsvm.wstar}")
    print(f"learned bias: {dsvm.bstar}")
    print(f"training accuracy: {np.mean(train_y == dsvm.predict(train_x))}")
    print(f"testing accuracy: {np.mean(test_y == dsvm.predict(test_x))}")

print("==== Dual SVM, b) ====")
Cs = [100/873, 500/873, 700/873]
gammas = [0.1, 0.5, 1, 5, 100]
sv = []
for C in Cs:
    for gamma in gammas:
        print(f"C = {C}")
        print(f"gamma = {gamma}")
        dsvm = svm.DualSVM(train_x, train_y, C=C, kernel='gaussian', gamma=gamma)
        print(f"learned weights: {dsvm.wstar}")
        print(f"learned bias: {dsvm.bstar}")
        print(f"number of support vectors: {len(dsvm.support)}")
        if C == 500/873: sv.append(dsvm.support)
        print(f"training accuracy: {np.mean(train_y == dsvm.predict(train_x, kernel='gaussian', gamma=gamma))}")
        print(f"testing accuracy: {np.mean(test_y == dsvm.predict(test_x, kernel='gaussian', gamma=gamma))}")

for i in range(4):
    count = 0
    for v in np.array(sv[i]):
        if v in np.array(sv[i+1]):
            count += 1
    print(f"overlap from gamma = {gammas[i]} to {gammas[i+1]}: {count}")