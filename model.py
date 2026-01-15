import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // ratio)
        self.fc2 = nn.Linear(channels // ratio, channels)

    def forward(self, x):
        avg = x.mean(dim=1)
        attn = torch.sigmoid(self.fc2(F.relu(self.fc1(avg))))
        return x * attn.unsqueeze(1)


class EEG_ASD_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(8, 32, 15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)

        self.ms3 = nn.Conv1d(32, 32, 3, padding=1)
        self.ms5 = nn.Conv1d(32, 32, 5, padding=2)
        self.ms7 = nn.Conv1d(32, 32, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)

        self.attn = ChannelAttention(32)
        self.dw = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.bn3 = nn.BatchNorm1d(32)

        self.pool = nn.AvgPool1d(4)
        self.lstm1 = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.bn2(self.ms3(x) + self.ms5(x) + self.ms7(x))
        x = self.attn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.elu(self.bn3(self.dw(x)))
        x = self.pool(x)
        x, _ = self.lstm1(x.permute(0, 2, 1))
        x, _ = self.lstm2(x)
        x = F.elu(self.fc1(x[:, -1]))
        return self.fc2(x)
