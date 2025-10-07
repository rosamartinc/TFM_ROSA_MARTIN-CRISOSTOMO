import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, downsample=None):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    # ResNet-1D con bloques básicos y GAP
    def __init__(self, n_classes, n_leads=12, target_len=1000, dropout=0.3,
                 base_channels=64, layers=(2,2,2), kernel_size=7, **kwargs):
        super().__init__()
        chs = [base_channels, base_channels*2, base_channels*4]
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        in_ch = base_channels
        stages = []
        for stage_idx, num_blocks in enumerate(layers):
            out_ch = chs[stage_idx]
            stride = 1 if stage_idx==0 else 2
            down = None
            if stride != 1 or in_ch != out_ch:
                down = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch)
                )
            blocks = [BasicBlock1D(in_ch, out_ch, kernel_size=kernel_size, stride=stride, downsample=down)]
            in_ch = out_ch
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock1D(in_ch, out_ch, kernel_size=kernel_size, stride=1, downsample=None))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(chs[len(layers)-1], n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x
