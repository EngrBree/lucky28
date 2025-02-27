import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ✅ Ensure script can access model architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP  # Import your model

# ✅ Define Dataset Class
class LotteryDataset(TensorDataset):
    def __init__(self, X, y):
        """Custom PyTorch dataset for lottery prediction."""
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        super().__init__(X, y)

# ✅ Load Data Function
def get_dataloaders(train_file, test_file, target_column, batch_size=64):
    """Loads train/test data, scales numerical features, and returns DataLoaders."""
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # ✅ Keep only numeric columns
    train = train.select_dtypes(include=['number'])
    test = test.select_dtypes(include=['number'])

    # ✅ Ensure target column exists
    if target_column not in train.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # ✅ Extract features & target
    X_train = train.drop(columns=[target_column]).values
    y_train = train[target_column].values.reshape(-1, 1)
    X_test = test.drop(columns=[target_column]).values
    y_test = test[target_column].values.reshape(-1, 1)

    # ✅ Scale numerical features (fit only on train set)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Do not fit again on test data

    # ✅ Create datasets
    train_dataset = LotteryDataset(X_train, y_train)
    test_dataset = LotteryDataset(X_test, y_test)

    # ✅ Create DataLoaders (⚠️ Fix: Set `num_workers=0` for Windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset, X_train, y_train

# ✅ Ensure script runs correctly on Windows
if __name__ == '__main__':  
    # ✅ Load dataset (for Big-Small)
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    target_column = "big_small_1"
    batch_size = 64

    train_loader, test_loader, train_dataset, X_train, y_train = get_dataloaders(train_file, test_file, target_column, batch_size)

    # ✅ Define Model, Loss, Optimizer, Scheduler
    input_dim = train_dataset.tensors[0].shape[1]  # Get input feature size
    model = MLP(input_dim=input_dim)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    # ✅ Early Stopping Variables
    best_val_loss = float('inf')
    patience = 7  # Number of epochs to wait before stopping
    patience_counter = 0

    # ✅ Cross-Validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # ✅ Train Model with Cross-Validation
    epochs = 50
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n===== Fold {fold + 1}/{k_folds} =====")
        
        # Split Data
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
        
        train_dataset = LotteryDataset(X_train_fold, y_train_fold)
        val_dataset = LotteryDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for epoch in range(epochs):
            # Training Loop
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).view(-1)
                loss = criterion(outputs, y_batch.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation Loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch).view(-1)
                    loss = criterion(outputs, y_batch.view(-1))
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Print Progress
            print(f"Epoch {epoch+1}, Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

            # ✅ Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "models/big_small_model.pth")  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered! Best validation loss: {best_val_loss:.4f}")
                    break  # Stop training if no improvement

            # ✅ Adjust Learning Rate using Scheduler
            scheduler.step(val_loss)

    # ✅ Load Best Model for Final Evaluation
    print("\n✅ Training Complete! Evaluating Model...")
    model.load_state_dict(torch.load("models/big_small_model.pth"))
    model.eval()

    # ✅ Evaluate on Test Set
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).view(-1)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch.view(-1)).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
