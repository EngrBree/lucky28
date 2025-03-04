import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ✅ Ensure script can access model architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP  # Import your model
from utils.focal_loss import FocalLoss

# ✅ Define Dataset & DataLoader
class LotteryDataset(TensorDataset):
    def __init__(self, file_path, target_column):
        data = pd.read_csv(file_path)
        data = data.select_dtypes(include=['number'])  # Drop non-numeric columns
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        X = data.drop(columns=[target_column]).values
        y = data[target_column].values.reshape(-1, 1)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        super().__init__(X, y)


def get_dataloaders(train_file, test_file, target_column, batch_size=32):
    train_dataset = LotteryDataset(train_file, target_column)
    test_dataset = LotteryDataset(test_file, target_column)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_dataset, test_dataset


# ✅ Load dataset
train_file = "data/train.csv"
test_file = "data/test.csv"
target_column = "big_small_1"
batch_size = 8

train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(train_file, test_file, target_column, batch_size)

# ✅ Define model, loss, optimizer
input_dim = train_dataset.tensors[0].shape[1]
model = MLP(input_dim=input_dim, dropout_rate=0.8)

criterion = FocalLoss()  # Binary classification loss
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-3)

# ✅ Adaptive Learning Rate Scheduling
onecycle_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=50)
plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)


# ✅ Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⛔ Early stopping triggered! Best validation loss: {self.best_loss:.4f}")
                return True
        return False


early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

# ✅ Train the model
epochs = 50
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).view(-1)
        loss = criterion(outputs, y_batch.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # ✅ Compute validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).view(-1)
            val_loss += criterion(outputs, y_batch.view(-1)).item()

    val_loss /= len(test_loader)

    # ✅ Print training progress
    print(f"Epoch {epoch+1} | Training Loss: {epoch_loss / len(train_loader):.4f} | Validation Loss: {val_loss:.4f}")

    # ✅ Step both schedulers
    onecycle_scheduler.step()
    plateau_scheduler.step(val_loss)

    # ✅ Early stopping
    if early_stopping(val_loss):
        break

    # ✅ Save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "models/best_big_small_model.pth")
        print("✅ Best model saved!")

# ✅ Save final model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/big_small_model.pth")
print("✅ Final Model Saved!")


# ✅ Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).view(-1)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == y_batch.view(-1)).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"🎯 Test Accuracy: {accuracy:.4f}")


# ✅ Run evaluation
evaluate_model(model, test_loader)
