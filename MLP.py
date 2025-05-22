import os
import json
import numpy as np
import imageio.v2 as imageio
from skimage.transform import resize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import tqdm


# -----------------------
# Improved Neural Network
# -----------------------
class ImprovedNeuralNet:
    def __init__(
        self,
        input_nodes,
        output_nodes,
        hidden_nodes_list=[],
        learning_rate=0.01,
        l2_reg=0.0001,
        class_weights=None,
    ):
        self.class_weights: np.ndarray = (
            class_weights if class_weights is not None else np.ones(output_nodes)
        )

        # Create a list of layer sizes: input, all hidden layers, then output
        self.layers = [input_nodes] + hidden_nodes_list + [output_nodes]
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        # Initialize the weight matrices in a list for each layer
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            # Initialize weights with random values and scale them using Xavier initialization
            n_in, n_out = self.layers[i], self.layers[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((n_out, 1))
            # Append the weights and biases to the lists
            self.weights.append(W)
            self.biases.append(b)

        # print the architecture of the neural network
        print("Neural Network Architecture:")
        for i in range(len(self.layers) - 1):
            print(f"Layer {i + 1}: {self.layers[i]} -> {self.layers[i + 1]}")
            print(f"Weight Matrix: {self.weights[i].shape}")
            print(f"Bias Vector: {self.biases[i].shape}")
        print(f"Input Layer: {self.layers[0]}")
        print(f"Hidden Layers: {self.layers[1:-1]}")
        print(f"Output Layer: {self.layers[-2]} -> {self.layers[-1]}")
        print(f"Learning Rate: {self.learning_rate:.4f}")

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        # Add numerical stability safeguards
        x = np.clip(x, -500, 500)  # Prevent extreme values
        x = np.nan_to_num(x)  # Replace any NaN values

        # Subtract max for numerical stability (you already have this)
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))

        # Add small epsilon to denominator to prevent division by zero
        return e_x / (e_x.sum(axis=0, keepdims=True) + 1e-10)

    def CE_loss(self, y_true, y_pred):
        # Cross-entropy loss function with class weighting
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calculate class weights based on inverse frequency
        if y_true.ndim == 1:
            y_true = np.eye(self.layers[-1])[y_true].T  # Convert to one-hot encoding
        # Ensure y_true is one-hot encoded
        if y_true.shape[0] != self.layers[-1]:
            raise ValueError(
                f"y_true should have shape ({self.layers[-1]}, n_samples), got {y_true.shape}"
            )
        class_indices = np.argmax(y_true, axis=0)

        # Apply weights to individual samples
        sample_weights = self.class_weights[class_indices]

        weighted_loss = -np.sum(
            sample_weights * np.sum(y_true * np.log(y_pred), axis=0)
        ) / np.sum(sample_weights)
        return weighted_loss

    # Train the neural network using backpropagation.
    def train(self, train_X, train_Y):
        """
        train_X: input_nodes x training_samples (features x samples)
        train_Y: output_nodes x training_samples (one-hot encoded)
        """
        # --- forward ---
        # Handle single sample case
        if len(train_X.shape) == 1:
            train_X = np.reshape(
                train_X, (-1, 1)
            )  # convert to column vector (features x 1)

        if len(train_Y.shape) == 1:
            train_Y = np.reshape(
                train_Y, (-1, 1)
            )  # convert to column vector (outputs x 1)

        # Correct dimension check for features x samples format
        assert (
            train_X.shape[0] == self.layers[0]
        ), f"Expected {self.layers[0]} features, got {train_X.shape[0]}"
        assert (
            train_Y.shape[0] == self.layers[-1]
        ), f"Expected {self.layers[-1]} output classes, got {train_Y.shape[0]}"

        X = train_X  # Shape: (input_nodes, batch_size)
        y = train_Y  # Shape: (output_nodes, batch_size)

        # Store the activations of each layer in a list
        activations = [X]
        zs = []

        # Forward pass through the network
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            # Compute the weighted sum of inputs for the next layer
            z = w.T @ X + b  # (n_out x n_in) @ (n_in x batch_size) + (n_out x 1)
            # Apply the relu activation function
            X = self.relu(z)
            # Store the activations
            activations.append(X)  # Shape: (n_out_l, batch_size)
            zs.append(z)

        # Compute the output layer
        z = self.weights[-1].T @ X + self.biases[-1]
        zs.append(z)
        output = self.softmax(z)
        activations.append(output)  # Add this line to store final output in activations

        # --- Compute loss ---
        prev_loss = self.CE_loss(y, output)

        # --- backward and updating weights & biases ---
        # Apply class weights (weighted cross-entropy derivative)
        # shape: (output_nodes, batch_size)

        # scale delta by class weights
        delta = (output - y) * self.class_weights.reshape(-1, 1)

        m = train_X.shape[1]

        # Backpropagate the error through all layers
        for l in reversed(range(len(self.weights))):
            # Calculate gradients for weights and biases
            dW = (
                activations[l] @ delta.T / m
            )  # Shape:  (n_in_l, batch_size) @ (batch_size, n_out_l) -> (n_in_l, n_out_l)
            # Add L2 regularization term
            dW += self.l2_reg * self.weights[l]
            db = np.sum(delta, axis=1, keepdims=True) / m  # Shape: (n_out_l, 1)

            # Update weights and biases
            self.weights[l] -= self.learning_rate * dW

            self.biases[l] -= self.learning_rate * db

            # Calculate delta for previous layer (if not at input layer)
            if l > 0:
                # Propagate error backward through weights and apply activation derivative
                delta = (self.weights[l] @ delta) * self.relu_derivative(zs[l - 1])

        return prev_loss

    def query(self, inputs_list: np.ndarray):
        if len(inputs_list.shape) == 1:
            assert inputs_list.shape[0] == self.layers[0]
            # Convert the input list into a column vector
            inputs_list = np.reshape(inputs_list, (-1, 1))

        X = inputs_list

        # Forward pass through the network
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            # Compute the weighted sum of inputs for the next layer
            z = w.T @ X + b  # (n_out x n_in) @ (n_in x batch_size) + (n_out x 1)
            # Apply the relu activation function
            X = self.relu(z)

        # Compute the output layer
        z = self.weights[-1].T @ X + self.biases[-1]
        output = self.softmax(z)

        return output


# -----------------------
# Utility Functions
# -----------------------
def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def preprocess_image(img_path, size=(64, 64)):
    img = imageio.imread(img_path, mode="L")
    img = resize(img, size, anti_aliasing=True)
    img = (img / 255.0 * 0.99) + 0.01
    return img.flatten()


def prepare_data(data_dict, base_path, count=None):
    """
    Outputs a numpy array of shape (n_features, n_samples) and a labels array of shape (n_samples,)
    where n_features is the number of pixels in the image

    """
    data = []
    labels = []
    if count is not None:
        data_dict = dict(list(data_dict.items())[:count])
    for rel_path, label in data_dict.items():
        full_path = os.path.normpath(os.path.join(base_path, rel_path))
        if not os.path.exists(full_path):
            print(f"❌ Missing: {full_path}")
            continue
        inputs = preprocess_image(full_path)
        data.append(inputs)
        labels.append(label)
    return (np.array(data).T), np.array(labels).T


def evaluate_model(model, X, y, label="Test", show_cm=True):
    y_pred = model.query(X)

    loss = model.CE_loss(y, y_pred)

    # add argmax to get the predicted class
    y_pred = np.argmax(y_pred, axis=0)

    y_true = y

    print(f"\n✅ Ran predictions for {count} images for {label} set")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n📊 {label} Results")
    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if show_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, cmap="Blues")
        plt.title(f"{label} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.show()


# -----------------------
# Main Execution
# -----------------------
BASE_DIR = "."  # Folder containing .json files and image folders

train_data = load_data(os.path.join(BASE_DIR, "train.json"))
val_data = load_data(os.path.join(BASE_DIR, "val.json"))
test_data = load_data(os.path.join(BASE_DIR, "test.json"))

INPUT_NODES = 4096
HIDDEN_SIZES = [1024, 256, 32]
N_CLASSES = 6
INITIAL_LR = 0.005
EPOCHS = 20
BATCH_SIZE = 500


# -----------------------
# Load the data from .npy files if they exist
# or prepare the data and save it to .npy files
# .npy files are faster to load
if os.path.exists(os.path.join(BASE_DIR, "X_train.npy")):
    print("📦 Loading data from .npy files...")
    X = np.load(os.path.join(BASE_DIR, "X_train.npy"))
    y = np.load(os.path.join(BASE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(BASE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(BASE_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"))
else:
    print("📦 Preparing data...")
    X, y = prepare_data(train_data, BASE_DIR)
    X_val, y_val = prepare_data(val_data, BASE_DIR)
    X_test, y_test = prepare_data(test_data, BASE_DIR)

    # save to .npy files
    np.save(os.path.join(BASE_DIR, "X_train.npy"), X)
    np.save(os.path.join(BASE_DIR, "y_train.npy"), y)
    np.save(os.path.join(BASE_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(BASE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(BASE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(BASE_DIR, "y_test.npy"), y_test)

print(f"✅ Loaded {X.shape[1]} training samples")
print(f"✅ Loaded {X_val.shape[1]} validation samples")
print(f"✅ Loaded {X_test.shape[1]} test samples")


# Get statistics of the dataset
print("\n📊 Dataset Statistics" "\n--------------------------------")
# distribution of classes in the training set
print("Training set class distribution:")
unique, counts = np.unique(y, return_counts=True)
for i, count in zip(unique, counts):
    print(f"Class {i}: {count} samples")
# distribution of classes in the validation set
print("\nValidation set class distribution:")
unique, counts = np.unique(y_val, return_counts=True)
for i, count in zip(unique, counts):
    print(f"Class {i}: {count} samples")
# distribution of classes in the test set
print("\nTest set class distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for i, count in zip(unique, counts):
    print(f"Class {i}: {count} samples")
# -------------------------


def compute_class_weights(y_labels, n_classes=6):
    class_counts = np.bincount(y_labels, minlength=n_classes)
    total = class_counts.sum()
    return total / (n_classes * class_counts)


class_weights = compute_class_weights(y, n_classes=N_CLASSES)
print(f"\nClass Weights: {class_weights}")

nn = ImprovedNeuralNet(
    input_nodes=INPUT_NODES,
    output_nodes=N_CLASSES,
    hidden_nodes_list=HIDDEN_SIZES,
    learning_rate=INITIAL_LR,
    class_weights=class_weights,
)
print("\n🧠 Starting training...")
for epoch in range(EPOCHS):
    current_lr = INITIAL_LR
    # # Apply learning rate decay
    current_lr = INITIAL_LR / (1 + 0.1 * epoch)  # Simple decay schedule
    nn.learning_rate = current_lr

    print(f"\nEpoch {epoch + 1}/{EPOCHS} (LR: {current_lr:.6f})")
    # shuffle the data
    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)
    X = X[:, indices]
    y = y[indices]

    # Split the data into batches
    num_batches = int(np.ceil(X.shape[1] / BATCH_SIZE))
    tqdm_iter = tqdm.tqdm(range(num_batches))
    # Convert labels to one-hot encoding
    one_hot_y = np.zeros((N_CLASSES, y.shape[0]))
    one_hot_y[y, np.arange(y.shape[0])] = 1
    for i in tqdm_iter:
        tqdm_iter.set_description(
            f"Epoch {epoch + 1}/{EPOCHS} - Batch {i + 1}/{num_batches}"
        )
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, X.shape[1])
        batch_X = X[:, start:end]
        batch_y = one_hot_y[:, start:end]
        # Train the model on the batch
        prev_loss = nn.train(batch_X, batch_y)
        tqdm_iter.set_postfix_str(f"Loss: {prev_loss:.4f}")

    evaluate_model(nn, X_val, y_val, label="Validation", show_cm=False)

evaluate_model(nn, X_test, y_test, label="Test")
