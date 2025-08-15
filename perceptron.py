import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Raw input data: [IQ, CGPA]
X_raw = np.array([
    [120, 8.5],
    [100, 7.0],
    [130, 9.0],
    [95, 6.5]
])

# Labels: 1 = Placed, 0 = Not Placed
y = np.array([1, 0, 1, 0])

# Normalize the data (important!)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0
lr = 0.1
epochs = 20

# Activation function
def step(z):
    return 1 if z >= 0 else 0

# Training loop
for epoch in range(epochs):
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        y_pred = step(z)
        error = y[i] - y_pred
        weights += lr * error * X[i]
        bias += lr * error

# Output trained values
print("Trained weights:", weights)
print("Trained bias:", bias)

# Predictions
print("\nPredictions:")
for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    y_pred = step(z)
    status = "Placed" if y_pred == 1 else "Not Placed"
    print(f"Student {i+1}: IQ = {X_raw[i][0]}, CGPA = {X_raw[i][1]} → {status}")

# --- Visualization ---

# Create a grid to plot decision boundary
x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_vals = -(weights[0] * x_vals + bias) / weights[1]

# Plot points
for i in range(len(X)):
    color = 'green' if y[i] == 1 else 'red'
    plt.scatter(X[i][0], X[i][1], color=color, label=f'Student {i+1}' if epoch == 0 else "")

# Plot decision boundary
plt.plot(x_vals, y_vals, color='blue', label='Decision Boundary')
plt.xlabel('Normalized IQ')
plt.ylabel('Normalized CGPA')
plt.title('Perceptron - Placement Prediction')
plt.legend()
plt.grid(True)
plt.show()

