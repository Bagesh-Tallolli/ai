import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# from sklearn.datasets import load_breast_cancer   # <-- Uncomment for Breast Cancer

# =====================================================
# =============== DATASET SELECTION ===================
# =====================================================

# -------- IRIS DATASET (ACTIVE) --------
iris = load_iris()
X = iris.data
feature_names = iris.feature_names


# -------- BREAST CANCER DATASET (COMMENTED) --------
"""
cancer = load_breast_cancer()
X = cancer.data
feature_names = cancer.feature_names
"""


def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# =====================================================
# ================= USER INPUT (k) ====================
# =====================================================

k = int(input("Enter number of clusters (k): "))

centroids, labels = kmeans(X, k)


# =====================================================
# ================= USER INPUT POINT ==================
# =====================================================

print("\nEnter a new data point:")
for i, name in enumerate(feature_names):
    print(f"{i+1}. {name}")

values = list(map(float, input(f"Enter {X.shape[1]} values separated by space: ").split()))
user_point = np.array(values)

# Find nearest centroid
distances = np.linalg.norm(centroids - user_point, axis=1)
cluster = np.argmin(distances)

print(f"\n✅ The given data point belongs to Cluster {cluster + 1}")


# =====================================================
# ==================== PLOTTING =======================
# =====================================================
# NOTE:
# For visualization, only the FIRST TWO FEATURES are plotted
# This works well for Iris
# For Breast Cancer, this is only a partial visualization

colors = ['r', 'g', 'b', 'c', 'm', 'y']

plt.figure(figsize=(8, 6))

for i in range(k):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        c=colors[i % len(colors)],
        label=f'Cluster {i+1}'
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='x', s=200, c='black', label='Centroids'
)

plt.scatter(
    user_point[0],
    user_point[1],
    marker='*', s=250, c='gold', label='User Input'
)

plt.title('K-Means Clustering')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

