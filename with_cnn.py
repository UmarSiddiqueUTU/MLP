import os
import numpy as np
import torch
from torch import nn as NN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import imageio
from skimage.transform import resize
import json
import tqdm
from torchvision import transforms

class CNNNet(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = NN.Sequential(
            NN.Conv2d(1, 32, kernel_size=3, padding=1),
            NN.ReLU(),
            NN.MaxPool2d(2),
            NN.Conv2d(32, 64, kernel_size=3, padding=1),
            NN.ReLU(),
            NN.MaxPool2d(2),
            NN.Conv2d(64, 128, kernel_size=3, padding=1),
            NN.ReLU(),
            NN.MaxPool2d(2),
        )
        self.fc_layers = NN.Sequential(
            NN.Linear(128 * 8 * 8, 256),
            NN.ReLU(),
            NN.Dropout(0.5),
            NN.Linear(256, 64),
            NN.ReLU(),
            NN.Dropout(0.5),
            NN.Linear(64, 6)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return x.argmax(dim=1)

def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def preprocess_image(img_path, size=(64, 64)):
    img = imageio.imread(img_path, mode="L")
    img = resize(img, size, anti_aliasing=True)
    img = img / 255.0
    return img.astype(np.float32)

def prepare_data(data_dict, base_path, count=None, augment=False):
    data = []
    labels = []
    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    if count is not None:
        data_dict = dict(list(data_dict.items())[:count])
    for rel_path, label in data_dict.items():
        full_path = os.path.normpath(os.path.join(base_path, rel_path))
        if not os.path.exists(full_path):
            print(f"❌ Missing: {full_path}")
            continue
        img = preprocess_image(full_path)
        if augment:
            img = (aug(img).squeeze().numpy() * 255).astype(np.float32) / 255.0
        data.append(img)
        labels.append(label)
    return np.stack(data), np.array(labels)

def evaluate_model(model, X, y, label="Test", show_cm=True):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        outputs = outputs.detach().cpu().numpy()
        y_pred = np.argmax(outputs, axis=1)
        y_true = y.cpu().numpy()

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

BASE_DIR = "."
train_data = load_data(os.path.join(BASE_DIR, "train.json"))
val_data = load_data(os.path.join(BASE_DIR, "val.json"))
test_data = load_data(os.path.join(BASE_DIR, "test.json"))

INPUT_SHAPE = (1, 64, 64)
N_CLASSES = 6
INITIAL_LR = 0.001  # Lower learning rate for Adam
EPOCHS = 20
BATCH_SIZE = 32

if os.path.exists(os.path.join(BASE_DIR, "X_train.npy")):
    print("📦 Loading data from .npy files...")
    X = np.load(os.path.join(BASE_DIR, "X_train.npy"))
    y = np.load(os.path.join(BASE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(BASE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(BASE_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"))
    X = X.T.reshape(-1, 1, 64, 64)
    X_val = X_val.T.reshape(-1, 1, 64, 64)
    X_test = X_test.T.reshape(-1, 1, 64, 64)
else:
    print("📦 Preparing data...")
    X, y = prepare_data(train_data, BASE_DIR, augment=True)
    X_val, y_val = prepare_data(val_data, BASE_DIR)
    X_test, y_test = prepare_data(test_data, BASE_DIR)
    np.save(os.path.join(BASE_DIR, "X_train.npy"), X.reshape(X.shape[0], -1).T)
    np.save(os.path.join(BASE_DIR, "y_train.npy"), y)
    np.save(os.path.join(BASE_DIR, "X_val.npy"), X_val.reshape(X_val.shape[0], -1).T)
    np.save(os.path.join(BASE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(BASE_DIR, "X_test.npy"), X_test.reshape(X_test.shape[0], -1).T)
    np.save(os.path.join(BASE_DIR, "y_test.npy"), y_test)
    X = X.reshape(-1, 1, 64, 64)
    X_val = X_val.reshape(-1, 1, 64, 64)
    X_test = X_test.reshape(-1, 1, 64, 64)

y = torch.from_numpy(y).long()
X = torch.from_numpy(X).float()
y_val = torch.from_numpy(y_val).long()
X_val = torch.from_numpy(X_val).float()
y_test = torch.from_numpy(y_test).long()
X_test = torch.from_numpy(X_test).float()

def compute_class_weights(y_labels, n_classes: int):
    y_labels = y_labels.numpy()
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

model = CNNNet()
optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

EPOCHS = 50  # Increase epochs for better training

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0
    tqdm_iter = tqdm.tqdm(range(0, X.shape[0], BATCH_SIZE), desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for i in tqdm_iter:
        end_idx = min(i + BATCH_SIZE, X.shape[0])
        inputs = X[i:end_idx]
        labels = y[i:end_idx]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        tqdm_iter.set_postfix(loss=loss.item())
    train_acc = correct / total
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_fn(val_outputs, y_val)
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
evaluate_model(model, X_test, y_test, label="Test", show_cm=True)
