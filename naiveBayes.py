import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# ===================== DATASET SELECTION ==================
# ==========================================================

# ---------------- IRIS DATASET ----------------------------
#iris = load_iris()
#X, y = iris.data, iris.target
#class_names = iris.target_names
#feature_names = iris.feature_names


# ---------------- CANCER DATASET --------------------------
# Uncomment this and comment Iris above to use Breast Cancer
# ---------------- CANCER DATASET (ONLY 4 FEATURES) --------
cancer = load_breast_cancer()

# Select only first 4 features
X = cancer.data[:, :4]
y = cancer.target

class_names = cancer.target_names
feature_names = cancer.feature_names[:4]


# ---------------- TITANIC DATASET -------------------------
# Uncomment this and comment others to use Titanic
# titanic = sns.load_dataset("titanic")
# titanic = titanic[['survived', 'pclass', 'sex', 'age', 'fare']]
# titanic.dropna(inplace=True)
#
# le = LabelEncoder()
# titanic['sex'] = le.fit_transform(titanic['sex'])
#
# X = titanic.drop('survived', axis=1).values
# y = titanic['survived'].values
# class_names = np.array(['Not Survived', 'Survived'])
# feature_names = ['pclass', 'sex', 'age', 'fare']


# ==========================================================
# ===================== NAIVE BAYES =========================
# ==========================================================

class NaiveBayes:
    def fit(self, X, y):
        self._classes = np.unique(y)
        self._mean = np.array([X[y == c].mean(axis=0) for c in self._classes])
        self._var = np.array([X[y == c].var(axis=0) for c in self._classes])
        self._priors = np.array([X[y == c].shape[0] / len(y) for c in self._classes])

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for idx, prior in enumerate(self._priors):
            posterior = np.log(prior)
            posterior += np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# ==========================================================
# ===================== TRAIN / TEST =======================
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

nb = NaiveBayes()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Accuracy:", np.mean(y_pred == y_test))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# ==========================================================
# ===================== USER INPUT =========================
# ==========================================================

print("\nEnter feature values for prediction")
print("Type 'q' to exit\n")

# -------- IRIS INPUT (4 FEATURES) -------------------------
# Uncomment when using Iris
#print("Iris Features:")
#for i, f in enumerate(feature_names):
#    print(f"{i+1}. {f}")

# -------- BREAST CANCER INPUT (30 FEATURES) ---------------
# Uncomment when using Cancer
print("Breast Cancer Features:")
for i, f in enumerate(feature_names):
     print(f"{i+1}. {f}")

# -------- TITANIC INPUT (4 FEATURES) ----------------------
# Uncomment when using Titanic
# print("Titanic Features:")
# for i, f in enumerate(feature_names):
#     print(f"{i+1}. {f}")

while True:
    try:
        user_input = input(f"\nEnter {X.shape[1]} values separated by space: ")

        if user_input.lower() == 'q':
            print("Exiting...")
            break

        values = list(map(float, user_input.split()))

        if len(values) != X.shape[1]:
            print(f"❌ Please enter exactly {X.shape[1]} values.")
            continue

        user_data = np.array([values])
        prediction = nb.predict(user_data)

        print("✅ Predicted class:", class_names[prediction[0]])

    except ValueError:
        print("❌ Invalid input. Enter numeric values only.")

