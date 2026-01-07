from sklearn.datasets import load_iris
# from sklearn.datasets import load_breast_cancer   # <-- Uncomment to use Breast Cancer
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# =============== DATASET SELECTION ===================
# =====================================================

# -------- IRIS DATASET (ACTIVE) --------
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names


# -------- BREAST CANCER DATASET (COMMENTED) --------
"""
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
class_names = ['Malignant', 'Benign']
"""


# =====================================================
# ================= TRAIN / TEST ======================
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]


# =====================================================
# ================= TRAIN MODEL =======================
# =====================================================

knn = KNN(k=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print('Accuracy: %.4f' % np.mean(y_pred == y_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# =====================================================
# ================= USER INPUT LOOP ===================
# =====================================================

print("\nEnter measurements to predict class")
print("Type 'q' to exit\n")

# -------- IRIS USER INPUT (ACTIVE) --------
print("Order:")
print("Sepal length, Sepal width, Petal length, Petal width (cm)")

"""
# -------- BREAST CANCER USER INPUT (COMMENTED) --------
print("Order:")
print("Mean radius, Mean texture, Mean perimeter, Mean area,")
print("Mean smoothness, Mean compactness, Mean concavity,")
print("Mean concave points, Mean symmetry, Mean fractal dimension")
"""

while True:
    try:
        user_input = input("\nEnter values separated by space: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            break

        values = list(map(float, user_input.split()))

        # ---- Check input size ----
        if len(values) != X.shape[1]:
            print(f"❌ Please enter exactly {X.shape[1]} values.\n")
            continue

        user_data = np.array([values])
        prediction = knn.predict(user_data)

        print("✅ Predicted class:", class_names[prediction[0]])

    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")

