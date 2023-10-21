from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
import gradient

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "./concrete/"

cc_train_x = []
cc_train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_train_x.append(terms_flt[:-1])
        cc_train_y.append(terms_flt[-1])

cc_test_x = []
cc_test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_test_x.append(terms_flt[:-1])
        cc_test_y.append(terms_flt[-1])

print("LMS with Batch Gradient Descent")
bgd, loss_bgd = gradient.batch_gradient_descent(cc_train_x, cc_train_y, learning_rate = 1e-3, epochs = 500)

print(f"weight vect: {bgd}")
print("LMS with Stochastic Gradient Descent")
sgd, loss_sgd = gradient.stochastic_gradient_descent(cc_train_x, cc_train_y, learning_rate = 1e-3, epochs = 500)

print(f"weight vect: {sgd}")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# ax.plot(np.array(range(len(loss_bgd))) * len(cc_train_x), loss_bgd, color = 'tab:blue', label = "batch")
ax.plot(loss_sgd, color = 'tab:orange', label = "stochastic")
ax.legend()
ax.set_title("Gradient Descent")
ax.set_xlabel("# of iterations")
ax.set_ylabel("Mean Squared Error")

plt.savefig("./out/error.png")
plt.clf()
print("LMS Analytic Method")
lms = gradient.linear_regression(cc_train_x, cc_train_y)

print(f"weight vect: {lms}")