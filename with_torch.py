import os
import numpy as np
import torch

from torch import nn as NN

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import imageio
from skimage.transform import resize
import json
import tqdm


class NeuralNetowrk(NN.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = NN.Sequential(
            NN.Linear(4096, 4096),
            NN.ReLU(),
            NN.Linear(4096, 1024),
            NN.ReLU(),
            NN.Linear(1024, 256),
            NN.ReLU(),
            NN.Linear(256, 32),
            NN.ReLU(),
            NN.Linear(32, 6)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return x.argmax(dim=1)


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


def evaluate_model(model, X, y, label="Test", show_cm=True):
    outputs = model(X)
    outputs = outputs.detach().numpy()
    y_pred = np.argmax(outputs, axis=1)
    y_true = y.detach().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n📊 {label} Results")
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


# -----------------------
# Main Execution
# -----------------------
BASE_DIR = "."  # Folder containing .json files and image folders

train_data = load_data(os.path.join(BASE_DIR, "train.json"))
val_data = load_data(os.path.join(BASE_DIR, "val.json"))
test_data = load_data(os.path.join(BASE_DIR, "test.json"))

INPUT_NODES = 4096
HIDDEN_SIZES = [4096, 1024, 256, 32]
N_CLASSES = 6
INITIAL_LR = 0.01
EPOCHS = 50
BATCH_SIZE = 32

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


# convert to torch loaders

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# Create the model
model = NeuralNetowrk()
optimizer = torch.optim.SGD(
    model.parameters(), lr=INITIAL_LR,
)

def compute_class_weights(y_labels, n_classes: int):
    y_labels = y_labels.detach().numpy()
    counts = {i: 0 for i in range(n_classes)}
    for label in y_labels:
        counts[label] += 1
    total = sum(counts.values())
    weights = np.zeros((n_classes, ))
    for i in range(n_classes):
        freq = counts.get(i, 1)
        weights[i] = total / (n_classes * freq)
    return weights

class_weights = compute_class_weights(y, n_classes=N_CLASSES)

print("Class weights:", class_weights)

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

loss = loss_fn(model(X.T), y)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    tqdm_iter = tqdm.tqdm(
        range(0, X.shape[1], BATCH_SIZE), desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )
    for i in tqdm_iter:
        # Dynamically handle the last batch
        end_idx = min(i + BATCH_SIZE, X.shape[1])
        inputs = X[:, i:end_idx].T  # Shape: (batch_size, 4096)
        labels = y[i:end_idx]  # Shape: (batch_size,)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        tqdm_iter.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.T)
        val_loss = loss_fn(val_outputs, y_val)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
        )

# confusion matrix for the test set
evaluate_model(model, X_test.T, y_test, label="Test", show_cm=True)

