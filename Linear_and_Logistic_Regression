import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression

# =========================
# LINEAR REGRESSION
# =========================

# Data
X = np.random.rand(100, 1) * 10
y = 2 * X + 3 + np.random.randn(100, 1) * 2

# sklearn model
sk_model = LinearRegression()
sk_model.fit(X, y)
sk_pred = sk_model.predict(X)

# PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Model
model = nn.Linear(1, 1)

# Loss + optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
epochs = 200
for epoch in range(epochs):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"[Linear] Epoch {epoch}, Loss: {loss.item()}")

# Visualization
with torch.no_grad():
    y_pred = model(X_tensor)

plt.scatter(X, y)
plt.plot(X, y_pred.numpy(), color="red")
plt.title("Linear Regression (PyTorch)")
plt.show()


# =========================
# LOGISTIC REGRESSION
# =========================

# Dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    flip_y=0.1,
    class_sep=0.8,
    random_state=42
)

# sklearn logistic regression
sk_log = LogisticRegression()
sk_log.fit(X, y)
sk_pred = sk_log.predict(X)

# Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Loss + optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
epochs = 300
for epoch in range(epochs):
    preds = model(X_tensor)
    loss = criterion(preds, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"[Logistic] Epoch {epoch}, Loss: {loss.item()}")

# Predictions
with torch.no_grad():
    probs = model(X_tensor)
    predicted = (probs > 0.5).float()

# Visualization predictions
plt.scatter(X[:, 0], X[:, 1], c=predicted.numpy())
plt.title("Logistic Regression (PyTorch)")
plt.show()

# Decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

with torch.no_grad():
    probs = model(grid_tensor)

Z = probs.numpy().reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Decision Boundary (PyTorch Logistic Regression)")
plt.show()

# Model parameters
print(model[0].weight)
print(model[0].bias)
