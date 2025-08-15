import numpy as np
import matplotlib.pyplot as plt

# Step activation
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron for AND
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                error = target - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def plot(self, X, y, title):
        plt.figure()
        for xi, target in zip(X, y):
            color = 'blue' if target == 1 else 'red'
            marker = 'x' if target == 1 else 'o'
            plt.scatter(xi[0], xi[1], c=color, marker=marker)

        x_vals = np.array([0, 1])
        if self.weights[1] != 0:
            y_vals = -(self.weights[0] * x_vals + self.bias) / self.weights[1]
            plt.plot(x_vals, y_vals, 'g-', label='Decision Boundary')
        else:
            plt.axvline(-self.bias / self.weights[0], color='g')

        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.grid(True)
        plt.legend()
        plt.show()

# Inputs and outputs for AND gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Train
model = Perceptron(input_size=2)
model.train(X, y)

# Predictions
print("AND Gate Predictions:")
for xi in X:
    print(f"{xi} -> {model.predict(xi)}")

# Plot
model.plot(X, y, "AND Gate - Decision Boundary")
