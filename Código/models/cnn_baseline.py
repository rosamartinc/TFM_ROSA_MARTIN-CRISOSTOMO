import torch, torch.nn as nn

class CNNBaseline1D(nn.Module):
    #Baseline 1D CNN sencillo 
    def __init__(self, n_classes, n_leads=12, target_len=1000, dropout=0.3, **kwargs):
        super().__init__()
        ch = 64
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, ch*2, 7, padding=3),   nn.BatchNorm1d(ch*2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch*2, ch*2, 7, padding=3), nn.BatchNorm1d(ch*2), nn.ReLU(), nn.MaxPool1d(2),
        )
        L = target_len // 8
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear((ch*2)*max(1, L), n_classes))
    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)
