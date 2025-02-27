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

        # ✅ Drop non-numeric columns (e.g., datetime)
        data = data.select_dtypes(include=['number'])  # Keeps only numerical columns
        
        # ✅ Ensure target column is included
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        X = data.drop(columns=[target_column]).values
        y = data[target_column].values.reshape(-1, 1)

        # ✅ Scale numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # ✅ Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        super().__init__(X, y)


def get_dataloaders(train_file, test_file, target_column, batch_size=64):
    train_dataset = LotteryDataset(train_file, target_column)
    test_dataset = LotteryDataset(test_file, target_column)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_dataset, test_dataset

# ✅ Load dataset (for Odd-Even)
train_file = "data/train.csv"
test_file = "data/test.csv"
target_column = "odd_even_1"
batch_size = 128

train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(train_file, test_file, target_column, batch_size)

# ✅ Define model, loss, optimizer
input_dim = train_dataset.tensors[0].shape[1]  # Get input feature size
model = MLP(input_dim=input_dim)
criterion = FocalLoss() # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


# ✅ Define Early Stopping
class EarlyStopping:
    """Stops training if validation loss does not improve after a given patience."""
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⛔ Early stopping triggered! Best validation loss: {self.best_loss:.4f}")
                return True  # Stop training
        return False


# ✅ Initialize Early Stopping & Learning Rate Scheduler
early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ✅ Initialize tracking variables
best_loss = float('inf')

# ✅ Train the model
epochs = 50
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

    # ✅ Step scheduler with validation loss
    scheduler.step(val_loss)

    # ✅ Early stopping check
    if early_stopping(val_loss):
        break  # Stop training

    # ✅ Save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pth")  
        print("✅ Best model saved!")


# ✅ Save final model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/odd_even_model.pth")
print("✅ Final Model Saved!")


# ✅ Evaluate the Model
def evaluate_model(model, test_loader):
    """Computes accuracy on the test set."""
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


# ✅ Run evaluation after training
evaluate_model(model, test_loader)