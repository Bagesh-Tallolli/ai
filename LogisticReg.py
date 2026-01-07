import matplotlib.pyplot as plt
import numpy as np
#from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer   # <-- Uncomment to use Breast Cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression(X, y, num_iterations=200, learning_rate=0.001):
    weights = np.zeros(X.shape[1])
    for _ in range(num_iterations):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient_val = np.dot(X.T, (h - y)) / y.shape[0]
        weights -= learning_rate * gradient_val
    return weights


# =====================================================
# =============== DATASET SELECTION ===================
# =====================================================

# -------- IRIS DATASET (ACTIVE) --------
"""
iris = load_iris()
X = iris.data[:, :2]  # sepal length & sepal width
y = (iris.target != 0) * 1  # binary classification
"""

# -------- BREAST CANCER DATASET (COMMENTED) --------

cancer = load_breast_cancer()
X = cancer.data[:, :2]   # mean radius & mean texture (for visualization)
y = cancer.target        # 0 = malignant, 1 = benign



# =====================================================
# ================= TRAIN / TEST ======================
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=9
)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

weights = logistic_regression(X_train_std, y_train)

y_pred = sigmoid(np.dot(X_test_std, weights)) > 0.5
print(f'Accuracy: {np.mean(y_pred == y_test):.4f}')


# =====================================================
# ================== DECISION PLOT ====================
# =====================================================

x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights)) > 0.5
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_std[:, 0], X_train_std[:, 1], c=y_train, alpha=0.8)
plt.title('Logistic Regression Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('plot.png')
plt.show()


# =====================================================
# ================= USER INPUT ========================
# =====================================================

print("\nEnter measurements to predict class")
"""
# -------- IRIS USER INPUT (ACTIVE) --------
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
user_input = np.array([[sepal_length, sepal_width]])
"""

# -------- BREAST CANCER USER INPUT (COMMENTED) --------
mean_radius = float(input("Mean radius: "))
mean_texture = float(input("Mean texture: "))
user_input = np.array([[mean_radius, mean_texture]])


user_input_std = sc.transform(user_input)

probability = sigmoid(np.dot(user_input_std, weights))
prob_value = probability.item()
prediction = int(prob_value >= 0.5)

print(f"\nPrediction probability: {prob_value:.4f}")

# -------- OUTPUT LABELS --------
"""
print(
    "Predicted class:",
    "Non-Setosa" if prediction == 1 else "Setosa"
)
"""

# For Breast Cancer:
print(
    "Predicted class:",
    "Benign" if prediction == 1 else "Malignant"
)

