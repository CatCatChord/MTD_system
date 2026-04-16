import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. ResNet 模块 ====================

class ResidualBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetTraffic64(nn.Module):
    """ResNet for 64x64x3 USTC图像"""
    def __init__(self, num_classes=20, layers=[2, 2, 2, 2], dropout=0.3):
        super().__init__()
        self.in_channels = 64
        # USTC输入是3通道64x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 64->32
        self.layer1 = self._make_layer(64, layers[0], stride=1)   # 32x32
        self.layer2 = self._make_layer(128, layers[1], stride=2)  # 16x16
        self.layer3 = self._make_layer(256, layers[2], stride=2)  # 8x8
        self.layer4 = self._make_layer(512, layers[3], stride=2)  # 4x4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers =[ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ==================== 2. LSTM 模块 ====================

class BiLSTMTrafficUSTC(nn.Module):
    """双向LSTM + 自注意力 for USTC [50, 15]"""
    def __init__(self, num_classes=20, input_size=15, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context)
        return out


# ==================== 3. CNN+RNN 模块 ====================

class CNNRNNTrafficUSTC(nn.Module):
    """CNN+RNN with Embedding for USTC [50, 256] int64"""
    def __init__(self, num_classes=20, vocab_size=256, embed_dim=64, seq_len=50, dropout=0.3):
        super().__init__()
        # Embedding层将0-255字节值映射到embed_dim维
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 1D-CNN处理
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(seq_len // 4)
        )
        
        # 双向LSTM
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, 
                           bidirectional=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x:[batch, seq_len=50, packet_bytes=256] - 可能是float或int
        # 转换为long类型用于Embedding
        if x.dtype != torch.long:
            x = x.long()
        
        # Embedding:[batch, seq_len, packet_bytes] -> [batch, seq_len, packet_bytes, embed_dim]
        x = self.embedding(x)  #[batch, 50, 256, 64]
        
        # 对每个包的embedding做平均，得到包级特征
        x = x.mean(dim=2)  # [batch, 50, 64]
        
        # CNN需要 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, 64, 50]
        x = self.cnn(x)  # [batch, 64, ~12]
        x = x.permute(0, 2, 1)  # [batch, ~12, 64]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  #[batch, ~12, 256]
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context)
        return out
# ==================== 4. CIC ResNet 模块 (28x28x1) ====================

class ResidualBlockCIC(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet28(nn.Module):
    """ResNet for CIC 28x28x1 images"""
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResidualBlockCIC(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlockCIC(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ==================== 5. CIC BiLSTM 模块 (50, 2) ====================

class BiLSTMCIC(nn.Module):
    """BiLSTM with Attention for CIC [50, 2]"""
    def __init__(self, num_classes, input_size=2, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, 
                           batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0)
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, 50, 256]
        attn_weights = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context)


# ==================== 6. CIC CNN+RNN 模块 V5 ====================

class CNNRNNCIC(nn.Module):
    """
    简化版CNN+RNN - 增加LayerNorm稳定输出，缓解震荡
    输入: [batch, 50, 256] float
    """
    def __init__(self, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        
        self.cnn_down = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(64, hidden, num_layers=2,
                           batch_first=True, bidirectional=True,
                           dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(hidden * 2)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, 50, 256]
        x = x.permute(0, 2, 1)  # [batch, 256, 50]
        x = self.cnn_down(x)  # [batch, 64, 50]
        
        x = x.permute(0, 2, 1)  # [batch, 50, 64]
        lstm_out, _ = self.lstm(x)  # [batch, 50, 256]
        
        lstm_out = self.layer_norm(lstm_out)
        
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return self.classifier(context)
