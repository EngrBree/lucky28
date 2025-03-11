import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)

        self.fc3 = nn.Linear(64, 32)
        self.norm3 = nn.LayerNorm(32)

        self.fc4 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.norm3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)
