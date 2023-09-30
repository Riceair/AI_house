import torch
import torch.nn as nn

class SingleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一層全連接層
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二層全連接層

    def forward(self, x):
        out = torch.relu(self.fc1(x))  # 使用ReLU作為激活函數
        out = self.fc2(out)
        return out