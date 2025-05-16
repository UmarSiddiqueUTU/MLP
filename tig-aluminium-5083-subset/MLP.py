import os
import json
import numpy as np
import imageio.v2 as imageio
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Neural Network Class
# ------------------------------
class NeuralNet:
    def __init__(self, input_nodes, output_nodes, hidden_nodes_list=[], learning_rate=0.01):
        self.layers = [input_nodes] + hidden_nodes_list + [output_nodes]
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            n_in, n_out = self.layers[i], self.layers[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((n_out, 1))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        activations = [inputs]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = w.T @ activations[-1] + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        deltas = [None] * len(self.weights)
        error = targets - activations[-1]
        deltas[-1] = error * activations[-1] * (1.0 - activations[-1])

        for i in range(len(self.weights) - 2, -1, -1):
            error = self.weights[i + 1] @ deltas[i + 1]
            deltas[i] = error * activations[i + 1] * (1.0 - activations[i + 1])

        for i in range(len(self.weights)):
            delta_w = self.learning_rate * (activations[i] @ deltas[i].T)
            self.weights[i] += delta_w
            self.biases[i] += self.learning_rate * deltas[i]

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        for w, b in zip(self.weights, self.biases):
            inputs = self.sigmoid(w.T @ inputs + b)
        return inputs

# ------------------------------
# Step 2: Preprocessing Functions
# ------------------------------
def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def preprocess_image(img_path, size=(64, 64)):
    img = imageio.imread(img_path, mode='L')
    img = resize(img, size, anti_aliasing=True)
    img = (img / 255.0 * 0.99) + 0.01
    return img.flatten()

# ------------------------------
# Step 3: Training the Network
# ------------------------------
BASE_DIR = 'tig-aluminium-5083-subset'
TRAIN_JSON_PATH = os.path.join(BASE_DIR, 'train.json')

train_data = load_data(TRAIN_JSON_PATH)

input_nodes = 64 * 64
hidden_nodes_list = [64, 32, 16]
output_nodes = 6
learning_rate = 0.01

nn = NeuralNet(input_nodes, output_nodes, hidden_nodes_list, learning_rate)

print("\nTraining started...")
for relative_path, label in train_data.items():
    full_path = os.path.normpath(os.path.join(BASE_DIR, relative_path))
    if not os.path.exists(full_path):
        print(f"Missing: {full_path}")
        continue

    try:
        inputs = preprocess_image(full_path)
        targets = np.zeros(output_nodes) + 0.01
        targets[int(label)] = 0.99
        nn.train(inputs, targets)
    except Exception as e:
        print(f"Error: {e}")
print("Training complete.")

# ------------------------------
# Step 4: Evaluating the Network
# ------------------------------
TEST_JSON_PATH = os.path.join(BASE_DIR, 'test.json')
test_data = load_data(TEST_JSON_PATH)

y_true, y_pred = [], []

print("\nEvaluating on test data...")
for relative_path, label in test_data.items():
    full_path = os.path.normpath(os.path.join(BASE_DIR, relative_path))
    if not os.path.exists(full_path):
        print(f"Missing: {full_path}")
        continue

    try:
        inputs = preprocess_image(full_path)
        outputs = nn.query(inputs)
        predicted = np.argmax(outputs)
        y_true.append(label)
        y_pred.append(predicted)
    except Exception as e:
        print(f"Error: {e}")

# ------------------------------
# Step 5: Metrics & Plot
# ------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"\n--- Test Results ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()



