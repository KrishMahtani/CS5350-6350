from os import makedirs
import ensemble

import matplotlib.pyplot as plt

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "./bank-4/"

def error(pred: list, target: list):
    assert len(pred) == len(target)
    mistakes = 0
    for i in range(len(pred)):
        if pred[i] != target[i]: mistakes += 1
    return mistakes / len(pred)

def HandleLine(line):
    terms = line.strip().split(",")
    t_dict = {
        "age": int(terms[0]),
        "job": terms[1],
        "marital": terms[2],
        "education": terms[3],
        "default": terms[4],
        "balance": int(terms[5]),
        "housing": terms[6],
        "loan": terms[7],
        "contact": terms[8],
        "day": int(terms[9]),
        "month": terms[10],
        "duration": int(terms[11]),
        "campaign": int(terms[12]),
        "pdays": int(terms[13]),
        "previous": int(terms[14]),
        "poutcome": terms[15],
        "label": terms[16]
    }
    return t_dict

if __name__ == '__main__':
    train_bank = []
    with open(dataset_loc + "train.csv", "r") as f:
        for line in f:
            train_bank.append(HandleLine(line))

    test_bank = []
    with open(dataset_loc + "test.csv", "r") as f:
        for line in f:
            test_bank.append(HandleLine(line))

    print("datasets loaded")

    ada = ensemble.AdaBoost()
    ada.train(train_bank, 5)

    print("Adaboost error in prediction:")

    train_pred = ada.predict(train_bank)
    print(error(train_pred, [d['label'] for d in train_bank]))

    test_pred = ada.predict(test_bank)
    print(error(test_pred, [d['label'] for d in test_bank]))

    print("running bagged trees:")
    x_pts = list(range(1,25)) + list(range(25,100,5)) + list(range(100, 550, 50))
    train_err = []
    test_err = []

    for x in x_pts:
        print(f"# trees: {x}")

        bag = ensemble.BaggedTrees()
        bag.train(train_bank, num_trees=x, num_samples=1000)

        train_pred = bag.predict(train_bank)
        train_err.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = bag.predict(test_bank)
        test_err.append(error(test_pred, [d['label'] for d in test_bank]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_pts, train_err, color = 'tab:blue', label = "training")
    ax.plot(x_pts, test_err, color = 'tab:orange', label = "testing")
    ax.legend()
    ax.set_title("Bagged Trees")
    ax.set_xlabel("# of trees")
    ax.set_ylabel("Misclassification Error")

    plt.savefig("./out/bagged_error.png")
    plt.clf()

    print("running random forest:")
    train_err_2 = []
    train_err_4 = []
    train_err_6 = []
    test_err_2 = []
    test_err_4 = []
    test_err_6 = []

    for x in x_pts:
        print(f"# trees: {x}")

        rf_2 = ensemble.RandomForest()
        rf_2.train(train_bank, num_trees=x, num_samples=1000, num_attributes=2)

        train_pred = rf_2.predict(train_bank)
        train_err_2.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = rf_2.predict(test_bank)
        test_err_2.append(error(test_pred, [d['label'] for d in test_bank]))

        rf_4 = ensemble.RandomForest()
        rf_4.train(train_bank, num_trees=x, num_samples=1000)

        train_pred = rf_4.predict(train_bank)
        train_err_4.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = rf_4.predict(test_bank)
        test_err_4.append(error(test_pred, [d['label'] for d in test_bank]))

        rf_6 = ensemble.RandomForest()
        rf_6.train(train_bank, num_trees=x, num_samples=1000, num_attributes=6)

        train_pred = rf_6.predict(train_bank)
        train_err_6.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = rf_6.predict(test_bank)
        test_err_6.append(error(test_pred, [d['label'] for d in test_bank]))

        

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_pts, train_err_2, label = "training, |G| = 2")
    ax.plot(x_pts, test_err_2, label = "testing, |G| = 2")
    ax.plot(x_pts, train_err_4, label = "training, |G| = 4")
    ax.plot(x_pts, test_err_4, label = "testing, |G| = 4")
    ax.plot(x_pts, train_err_6, label = "training, |G| = 6")
    ax.plot(x_pts, test_err_6, label = "testing, |G| = 6")
    ax.legend()
    ax.set_title("Random Forest")
    ax.set_xlabel("# of trees")
    ax.set_ylabel("Misclassification Error")

    plt.savefig("./out/randomforest_error.png")
    plt.clf()