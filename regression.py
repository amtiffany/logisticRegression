import struct
import numpy as np
from os.path import join
from sklearn.metrics import classification_report


def load_images(filepath: str):
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float64)


def load_labels(filepath: str):
    with open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def sigmoid(z):
    # 1/(1 + e^a), a is an element of an np array z
    return 1 / (1 + np.exp(-z))


def fit(X, Y, lr, iterations):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features, dtype = np.float64)
    bias = 0

    for i in range(iterations):
        model = np.dot(X, weights) + bias
        predictions = sigmoid(model)

        dw = (1 / num_samples) * np.dot(X.T, (predictions - Y))
        db = (1 / num_samples) * np.sum(predictions - Y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

    
# load data
input_path = "mnist_input"
training_images_filepath = join(input_path, "train-images.idx3-ubyte")
training_labels_filepath = join(input_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels.idx1-ubyte")

X_train = load_images(training_images_filepath) / 255.0
y_train = load_labels(training_labels_filepath)
X_test = load_images(test_images_filepath) / 255.0
y_test = load_labels(test_labels_filepath)

target_digit = 4
y_train_binary = (y_train == target_digit).astype(np.float64)
y_test_binary = (y_test == target_digit).astype(np.float64)

# Train model
weights, bias = fit(X_train, y_train_binary, lr=0.1, iterations=1000)

# Predict on test set
y_pred = (predict(X_test, weights, bias) >= 0.5).astype(np.int64)

# print report
print("\nClassification Report (1 = target digit, 0 = not):")
print(classification_report(y_test_binary, y_pred))