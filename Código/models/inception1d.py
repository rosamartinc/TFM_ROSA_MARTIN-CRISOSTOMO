import torch
import torch.nn as nn

class InceptionModule1D(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=32, kernel_sizes=(9,19,39), use_residual=True):
        super().__init__()
        self.use_bottleneck = in_ch > bottleneck
        bottleneck_ch = bottleneck if self.use_bottleneck else in_ch
        self.bottleneck = nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()

        convs = []
        for k in kernel_sizes:
            convs.append(nn.Conv1d(bottleneck_ch, out_ch, kernel_size=k, padding=k//2, bias=False))
        self.convs = nn.ModuleList(convs)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch * (len(kernel_sizes)+1))
        self.relu = nn.ReLU(inplace=True)
        self.use_residual = use_residual
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch * (len(kernel_sizes)+1), kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch * (len(kernel_sizes)+1))
            )
        else:
            self.residual = None

    def forward(self, x):
        z = self.bottleneck(x) if self.use_bottleneck else x
        y = [conv(z) for conv in self.convs]
        y.append(self.pool_conv(self.maxpool(x)))
        y = torch.cat(y, dim=1)
        y = self.bn(y)
        if self.residual is not None:
            y = self.relu(y + self.residual(x))
        else:
            y = self.relu(y)
        return y

class Inception1D(nn.Module):
    # InceptionTime-like 1D con ramas multi-kernel y residual
    def __init__(self, n_classes, n_leads=12, target_len=1000, dropout=0.3,
                 n_blocks=6, out_ch=32, bottleneck=32, kernel_sizes=(9,19,39), **kwargs):
        super().__init__()
        blocks = []
        in_ch = n_leads
        for _ in range(n_blocks):
            blocks.append(InceptionModule1D(in_ch, out_ch, bottleneck=bottleneck, kernel_sizes=kernel_sizes, use_residual=True))
            in_ch = out_ch * (len(kernel_sizes)+1)
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
