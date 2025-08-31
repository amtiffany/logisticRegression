import struct
import numpy as np
from os.path import join
from sklearn.metrics import classification_report


def load_images(filepath):
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float64)


def load_labels(filepath):
    with open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def sigmoid(z):
    # 1/(1 + e^z), z is an element of an np array Z
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



# ~~~~~~~~ Single Digit Classification ~~~~~~~~
    
# load data
input_path = "mnist_input"
training_images_filepath = join(input_path, "train-images.idx3-ubyte")
training_labels_filepath = join(input_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels.idx1-ubyte")

X_train = load_images(training_images_filepath) / 255
y_train = load_labels(training_labels_filepath)
X_test = load_images(test_images_filepath) / 255
y_test = load_labels(test_labels_filepath)

target_digit = 2
y_train_binary = (y_train == target_digit).astype(np.float64)
y_test_binary = (y_test == target_digit).astype(np.float64)

# Train model
print(f"Training binary model for digit '{target_digit}'...")
weights, bias = fit(X_train, y_train_binary, lr=0.1, iterations=1000)

# Predict on test set
y_pred = (predict(X_test, weights, bias) >= 0.5).astype(np.int64)

# print report
print("\nClassification Report (1 = target digit):")
print(classification_report(y_test_binary, y_pred))


'''
Sample output:

Training binary model for digit 4..."

Classification Report (1 = target digit):
              precision    recall  f1-score   support

         0.0       0.98      0.99      0.99      9018
         1.0       0.92      0.84      0.88       982

    accuracy                           0.98     10000
   macro avg       0.95      0.92      0.93     10000
weighted avg       0.98      0.98      0.98     10000

'''


# ~~~~~~~~ Multi Digit Classification ~~~~~~~~

# Train 10 binary classifiers
models = []
for digit in range(10):
    print(f"Training model for digit {digit}...")
    y_train_binary = (y_train == digit).astype(np.float64)
    weights, bias = fit(X_train, y_train_binary, lr=0.1, iterations=1000)
    models.append((weights, bias))

# pick the digit with the highest probability
all_probs = []
for (weights, bias) in models:
    probs = predict(X_test, weights, bias)
    all_probs.append(probs)

all_probs = np.array(all_probs)          # shape (10, num_samples)
y_pred = np.argmax(all_probs, axis=0)    # highest probability per sample

# print report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


'''
Sample output:

Training model for digit 0...
Training model for digit 1...
Training model for digit 2...
Training model for digit 3...
Training model for digit 4...
Training model for digit 5...
Training model for digit 6...
Training model for digit 7...
Training model for digit 8...
Training model for digit 9...

Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.98      0.96       980
           1       0.95      0.97      0.96      1135
           2       0.91      0.86      0.89      1032
           3       0.88      0.90      0.89      1010
           4       0.90      0.92      0.91       982
           5       0.90      0.80      0.85       892
           6       0.92      0.95      0.93       958
           7       0.91      0.90      0.90      1028
           8       0.83      0.88      0.85       974
           9       0.87      0.86      0.87      1009

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

'''