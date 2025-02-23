#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import json

#############################
# Model Definition (Matches Training)
#############################
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1)  # Raw logits for BCEWithLogitsLoss
        self.dropout = nn.Dropout(0.2)  # Reduced dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

#############################
# Data Loading for Testing
#############################
def load_test_data(test_path, target_column):
    data = pd.read_csv(test_path)
    print("Test CSV Columns:", data.columns.tolist())
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values.reshape(-1, 1)
    # Re-scale to ensure consistency (if necessary)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def create_test_loader(X, y, batch_size=64):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

#############################
# Evaluation Functions
#############################
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, total = 0, 0, 0
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch.view(-1))
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            preds_list.extend(predicted.cpu().numpy())
            targets_list.extend(y_batch.view(-1).cpu().numpy())
            correct += (predicted == y_batch.view(-1)).sum().item()
            total += y_batch.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / float(total)
    print(f"\nOverall Test Loss: {avg_loss:.4f}, Overall Test Accuracy: {accuracy:.4f}")
    return np.array(preds_list), np.array(targets_list), avg_loss, accuracy

def group_accuracy(preds, targets, group_size=10):
    n = len(preds)
    groups = [np.mean(preds[i:i+group_size] == targets[i:i+group_size])
              for i in range(0, n, group_size)]
    return groups

def comprehensive_evaluation(model, test_loader, target_column, group_size=10):
    preds, targets, avg_loss, overall_acc = evaluate_model(model, test_loader)
    groups = group_accuracy(preds, targets, group_size)
    
    # Prepare results dictionary
    results = {
        "overall_test_loss": avg_loss,
        "overall_test_accuracy": overall_acc,
        "group_accuracies": groups,
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "classification_report": classification_report(targets, preds, output_dict=True)
    }
    
    # Terminal display: truncate group output if too many groups
    max_display = 10
    if len(groups) > max_display:
        display_groups = groups[:5] + ["..."] + groups[-5:]
    else:
        display_groups = groups
    print("\nGroup-wise Accuracy (per 10 draws):")
    for i, acc in enumerate(display_groups, start=1):
        if isinstance(acc, str):
            print(f"Group {i}: {acc}")
        else:
            print(f"Group {i}: {acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))
    print("\nClassification Report:")
    print(classification_report(targets, preds))
    
    # Save full results to separate JSON files based on target column
    filename = f"evaluation_results_{target_column}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nFull evaluation results saved to {filename}")
    
    return results

def load_and_evaluate(model_path, test_path, target_column, batch_size=64):
    X, y = load_test_data(test_path, target_column)
    test_loader = create_test_loader(X, y, batch_size)
    input_dim = X.shape[1]
    model = MLP(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"\n--- Evaluating Model from {model_path} for target '{target_column}' ---")
    comprehensive_evaluation(model, test_loader, target_column, group_size=10)

#############################
# Main Evaluation Execution
#############################
if __name__ == "__main__":
    import json
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("=== Evaluating Odd/Even Model ===")
    load_and_evaluate("model_odd_even_trained.pth", "test.csv", "odd_even_1", batch_size=64)
    
    print("\n=== Evaluating Big/Small Model ===")
    load_and_evaluate("model_big_small_trained.pth", "test.csv", "big_small_1", batch_size=64)
    


