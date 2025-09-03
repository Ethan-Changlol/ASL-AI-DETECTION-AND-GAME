import pandas as pd
import string
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

# Load normalized dataset
df = pd.read_csv("hand_landmarks_normalized.csv", header=None)

# Balance dataset (100 samples per class, oversample if needed)
balanced = []
for c in string.ascii_uppercase:
    samples = df[df[0] == c]
    if len(samples) > 100:
        samples = samples.sample(100, random_state=0)
    elif len(samples) > 0:
        samples = resample(samples, replace=True, n_samples=100, random_state=0)
    balanced.append(samples)

df = pd.concat(balanced)

# Features/labels
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Train neural net
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=0)
clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# Save model
dump(clf, "asl.joblib")
print("✅ Model saved as asl.joblib")
