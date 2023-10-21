import numpy as np

def mean_squared_error(predictions, targets) -> float:
    assert len(predictions) == len(targets)
    predictions, targets = np.array(predictions), np.array(targets)
    return np.sum((targets - predictions) ** 2) / (2 * len(targets))

class LinearRegressionModel:
    def __init__(self, weights: list):
        self.weights = weights

    def __str__(self) -> str:
        return str(self.weights)

    def predict(self, data) -> list:
        return [np.dot(self.weights, d) for d in data]

def batch_gradient_descent(data, target, learning_rate: float = 0.01, epochs: int = 100, convergence_threshold = 1e-6):
    weights = np.ones_like(data[0])
    losses, last_loss, diff = [], 9999, 1

    for ep in range(epochs):
        if diff <= convergence_threshold:
            break

        gradient = np.zeros_like(weights)
        for j in range(len(gradient)):
            for xi, yi in zip(data, target):
                gradient[j] -= (yi - np.dot(weights, xi)) * xi[j]

        weights = weights - learning_rate * gradient

        loss = mean_squared_error(target, [np.dot(weights, xi) for xi in data])
        diff = abs(loss - last_loss)
        last_loss = loss
        losses.append(loss)

    print(f"Converged at epoch {ep} with a change of {diff}")
    return LinearRegressionModel(weights), losses

def stochastic_gradient_descent(data, target, learning_rate: float = 0.01, epochs: int = 100, convergence_threshold = 1e-6):
    weights = np.ones_like(data[0])
    losses, last_loss, diff = [], 9999, 1

    for ep in range(epochs):
        if diff <= convergence_threshold:
            break

        for i in range(len(data)):
            xi, yi = data[i], target[i]
            gradient = np.zeros_like(weights)
            for j in range(len(gradient)):
                gradient[j] = (yi - np.dot(weights, xi)) * xi[j]

            weights = weights + learning_rate * gradient

            loss = mean_squared_error(target, [np.dot(weights, xi) for xi in data])
            diff = abs(loss - last_loss)
            last_loss = loss
            losses.append(loss)

    print(f"Converged at epoch {ep} with a change of {diff}")
    return LinearRegressionModel(weights), losses

def linear_regression(data, target):
    data = np.transpose(np.array(data))
    target = np.array(target)
    weights = np.linalg.inv(data @ np.transpose(data)) @ (data @ target)
    return LinearRegressionModel(weights)