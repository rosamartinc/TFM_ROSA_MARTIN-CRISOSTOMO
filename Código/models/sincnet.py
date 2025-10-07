import torch, torch.nn as nn, torch.nn.functional as F

class SincConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate, in_channels=1, min_low_hz=0.0, min_band_hz=1.0):
        super().__init__()
        if in_channels != 1: raise ValueError('SincConv1d espera entrada mono-canal.')
        self.out_channels, self.kernel_size, self.sample_rate = out_channels, kernel_size, sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        self.low_hz_  = nn.Parameter(torch.randn(out_channels) * 0.1 + 30.0)
        self.band_hz_ = nn.Parameter(torch.randn(out_channels) * 0.1 + 100.0)
        n_lin = torch.linspace(0, self.kernel_size - 1, steps=self.kernel_size)
        self.register_buffer('n_lin', n_lin)
        self.register_buffer('window', torch.hamming_window(self.kernel_size, periodic=False))
        self.register_buffer('center', torch.tensor((self.kernel_size - 1) / 2.0))
    def forward(self, x):
        low  = F.softplus(self.low_hz_)  + self.min_low_hz
        band = F.softplus(self.band_hz_) + self.min_band_hz
        high = torch.clamp(low + band, max=(self.sample_rate / 2 - 1.0))
        t_right = (self.n_lin - self.center + 1e-9) / self.sample_rate
        filters = []
        for i in range(self.out_channels):
            low_hz, high_hz = low[i], high[i]
            bandpass = (2*high_hz)*torch.sinc(2*high_hz*t_right) - (2*low_hz)*torch.sinc(2*low_hz*t_right)
            bandpass = bandpass * self.window
            bandpass = bandpass / (bandpass.abs().sum() + 1e-9)
            filters.append(bandpass)
        kernel = torch.stack(filters).unsqueeze(1)
        return F.conv1d(x, kernel, stride=1, padding=self.kernel_size // 2, groups=1)

class LeadWiseSincBlock(nn.Module):
    def __init__(self, n_leads=12, sinc_out=16, sinc_kernel=251, sampling_rate=100):
        super().__init__()
        self.n_leads = n_leads
        self.sincs = nn.ModuleList([
            SincConv1d(out_channels=sinc_out, kernel_size=sinc_kernel, sample_rate=sampling_rate, in_channels=1)
            for _ in range(n_leads)
        ])
        self.bn = nn.BatchNorm1d(n_leads * sinc_out)
        self.mix = nn.Conv1d(n_leads * sinc_out, 64, kernel_size=1)
        self.bn_mix = nn.BatchNorm1d(64)
    def forward(self, x):
        outs = [self.sincs[c](x[:, c:c+1, :]) for c in range(self.n_leads)]
        y = torch.cat(outs, dim=1)
        y = torch.relu(self.bn(y))
        y = torch.relu(self.bn_mix(self.mix(y)))
        return y

class SincNetECG(nn.Module):
    def __init__(self, n_classes, n_leads=12, sampling_rate=100, target_len=1000, dropout=0.3, **kwargs):
        super().__init__()
        self.front = LeadWiseSincBlock(n_leads=n_leads, sinc_out=16, sinc_kernel=251, sampling_rate=sampling_rate)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        L = target_len // 16
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128 * max(1, L), n_classes)
    def forward(self, x):
        x = self.front(x)
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.head(x)
