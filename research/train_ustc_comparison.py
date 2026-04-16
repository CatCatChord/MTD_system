#!/usr/bin/env python3
"""
USTC-TFC2016 模型横向对比训练脚本
==================================
实现三个模型（ResNet/LSTM/CNN+RNN）的公平对比
统一要素：
1. 加权交叉熵损失（动态计算类别权重）
2. 多维评估体系（Accuracy/Precision/Recall/Macro-F1）
3. 以F1为导向的早停机制
4. 固定随机种子保证可复现

USTC数据格式：
- ResNet: [3, 64, 64] - 3通道64x64图像
- LSTM: [50, 15] - 50时间步，15维特征(3时序+12统计)
- CNN+RNN: [50, 256] int64 - 用于nn.Embedding
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
BASE_DIR = Path(r"/home/ckb/malicious_traffic_detection/research")
DATA_DIR = Path(r"/mnt/data/processed_data_fixed")  # USTC数据路径
SAVE_DIR = BASE_DIR / "comparison_results_ustc"
SAVE_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ==================== 1. 固定随机种子 ====================

def seed_everything(seed=42):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[✓] 随机种子已固定: {seed}")

# ==================== 2. 模型定义（适配USTC格式） ====================

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
    def __init__(self, num_classes, layers=[2, 2, 2, 2], dropout=0.3):
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
        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
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


class BiLSTMTrafficUSTC(nn.Module):
    """双向LSTM + 自注意力 for USTC [50, 15]"""
    def __init__(self, num_classes, input_size=15, hidden_dim=128, num_layers=2, dropout=0.3):
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


class CNNRNNTrafficUSTC(nn.Module):
    """CNN+RNN with Embedding for USTC [50, 256] int64"""
    def __init__(self, num_classes, vocab_size=256, embed_dim=64, seq_len=50, dropout=0.3):
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
        # x: [batch, seq_len=50, packet_bytes=256] - 可能是float或int
        # 转换为long类型用于Embedding
        if x.dtype != torch.long:
            x = x.long()
        
        # Embedding: [batch, seq_len, packet_bytes] -> [batch, seq_len, packet_bytes, embed_dim]
        x = self.embedding(x)  # [batch, 50, 256, 64]
        
        # 对每个包的embedding做平均，得到包级特征
        x = x.mean(dim=2)  # [batch, 50, 64]
        
        # CNN需要 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, 64, 50]
        x = self.cnn(x)  # [batch, 64, ~12]
        x = x.permute(0, 2, 1)  # [batch, ~12, 64]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, ~12, 256]
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context)
        return out


# ==================== 3. 数据加载（USTC格式） ====================

def load_data_ustc(model_type, batch_size=128, val_split=0.1):
    """加载USTC数据"""
    data_file = DATA_DIR / f"USTC_{model_type}_fixed.npz"
    
    if not data_file.exists():
        raise FileNotFoundError(f"USTC数据文件不存在: {data_file}\n请先运行 process_ustc.py 预处理数据")
    
    print(f"\n加载数据: {data_file}")
    data = np.load(data_file, allow_pickle=True)

    X_train = torch.FloatTensor(data['X_train'])
    X_test = torch.FloatTensor(data['X_test'])
    y_train = torch.LongTensor(data['y_train'])
    y_test = torch.LongTensor(data['y_test'])
    
    if 'classes' in data:
        class_names = data['classes'].tolist()
        if class_names and isinstance(class_names[0], bytes):
            class_names = [c.decode('utf-8') for c in class_names]
    else:
        class_names = ['Benign', 'Malware']

    num_classes = int(np.unique(y_train).size)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}, 类别数: {num_classes}")
    print(f"类别名称: {class_names}")

    # 计算类别权重
    class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
    print(f"类别分布: {dict(zip(range(num_classes), class_counts))}")
    
    # 验证集划分
    val_loader = None
    if 0.0 < val_split < 1.0:
        val_n = int(len(X_train) * val_split)
        idx = torch.randperm(len(X_train))
        train_idx, val_idx = idx[val_n:], idx[:val_n]
        
        train_dataset = TensorDataset(X_train[train_idx], y_train[train_idx])
        val_dataset = TensorDataset(X_train[val_idx], y_train[val_idx])
        
        # 加权采样器
        train_counts = np.bincount(y_train[train_idx].numpy())
        train_weights = 1.0 / (train_counts + 1e-8)
        sample_weights = train_weights[y_train[train_idx].numpy()]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
        print(f"验证集: {val_n} 条样本")
    else:
        train_counts = class_counts
        train_weights = 1.0 / (train_counts + 1e-8)
        sample_weights = train_weights[y_train.numpy()]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, 
                                 sampler=sampler, num_workers=4, pin_memory=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes, class_names


# ==================== 4. 统一评估函数 ====================

def evaluate_model(model, loader, criterion, device):
    """
    统一评估函数 - 返回多维指标
    Returns: loss, accuracy, precision, recall, f1_macro, predictions, labels
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    
    # sklearn计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1_macro, all_preds, all_labels


# ==================== 5. 训练函数 ====================

def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        if batch_idx % 20 == 0:  # USTC数据量小，打印频率更高
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, f1_macro


def train_model_ustc(model_type, batch_size=128, epochs=50, lr=1e-3, 
                     patience=10, seed=42):
    """
    统一训练函数 - 所有模型使用相同的训练机制
    """
    print(f"\n{'='*70}")
    print(f"训练 USTC {model_type.upper()}")
    print(f"{'='*70}")
    
    # 固定随机种子
    seed_everything(seed)
    
    # 加载数据
    train_loader, val_loader, test_loader, num_classes, class_names = \
        load_data_ustc(model_type, batch_size=batch_size)
    
    # 创建模型
    if model_type == 'resnet':
        model = ResNetTraffic64(num_classes).to(DEVICE)
    elif model_type == 'lstm':
        sample = next(iter(train_loader))[0]
        input_size = sample.shape[-1]
        model = BiLSTMTrafficUSTC(num_classes, input_size=input_size).to(DEVICE)
    elif model_type == 'cnn_rnn':
        model = CNNRNNTrafficUSTC(num_classes).to(DEVICE)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 统一加权交叉熵损失
    class_counts = np.bincount([label.item() for _, labels in train_loader for label in labels], 
                               minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"类别权重: {weights.cpu().numpy().round(4)}")
    
    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # 训练记录
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [],
        'test_recall': [], 'test_f1': [], 'lr': []
    }
    
    # 早停机制 - 以F1为导向
    best_val_f1 = 0.0
    trigger_times = 0
    best_epoch = 0
    
    print(f"\n开始训练 (早停耐心值: {patience})...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        
        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = \
            evaluate_model(model, val_loader, criterion, DEVICE)
        
        # 测试
        test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = \
            evaluate_model(model, test_loader, criterion, DEVICE)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(test_precision)
        history['test_recall'].append(test_recall)
        history['test_f1'].append(test_f1)
        history['lr'].append(current_lr)
        
        # 打印进度
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, P: {test_precision:.4f}, R: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # 学习率调度
        scheduler.step(val_f1)
        
        # 以F1为导向保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            trigger_times = 0
            model_path = SAVE_DIR / f"best_{model_type}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'test_f1': test_f1,
                'num_classes': num_classes,
                'class_names': class_names
            }, model_path)
            print(f"  ✓ 最佳模型已保存 (Val F1: {val_f1:.4f})")
        else:
            trigger_times += 1
            print(f"  F1未提升 ({trigger_times}/{patience})")
        
        # 早停检查
        if trigger_times >= patience:
            print(f"\n[!] 连续 {patience} 轮F1无提升，触发早停!")
            break
    
    # 加载最佳模型进行最终评估
    print(f"\n{'='*70}")
    print("最终评估 (使用最佳模型)")
    print(f"{'='*70}")
    
    checkpoint = torch.load(SAVE_DIR / f"best_{model_type}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_precision, test_recall, test_f1, preds, labels = \
        evaluate_model(model, test_loader, criterion, DEVICE)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    print(f"\n最佳模型来自 Epoch {checkpoint['epoch']+1}")
    print(f"测试集指标:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # 保存结果
    results = {
        'model_type': model_type,
        'best_epoch': int(checkpoint['epoch'] + 1),
        'best_val_f1': float(best_val_f1),
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1)
        },
        'history': history,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    with open(SAVE_DIR / f"results_{model_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].plot(history['test_acc'], label='Test')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history['train_f1'], label='Train')
    axes[0, 2].plot(history['val_f1'], label='Val')
    axes[0, 2].plot(history['test_f1'], label='Test')
    axes[0, 2].set_title('F1-Score (Macro)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].set_title('Val Precision & Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['test_precision'], label='Precision')
    axes[1, 1].plot(history['test_recall'], label='Recall')
    axes[1, 1].set_title('Test Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(history['lr'])
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"curves_{model_type}.png", dpi=150)
    plt.close()
    
    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'USTC {model_type.upper()} Confusion Matrix (Normalized)',
           ylabel='True Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    plt.savefig(SAVE_DIR / f"confusion_{model_type}.png", dpi=150)
    plt.close()
    
    print(f"\n[✓] USTC {model_type.upper()} 训练完成!")
    print(f"  最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch+1})")
    print(f"  最终测试F1: {test_f1:.4f}")
    
    return results


# ==================== 6. 对比汇总 ====================

def compare_results():
    """对比三个模型的结果"""
    print("\n" + "="*70)
    print("USTC 模型横向对比汇总")
    print("="*70)
    
    results = {}
    for model_type in ['resnet', 'lstm', 'cnn_rnn']:
        result_file = SAVE_DIR / f"results_{model_type}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[model_type] = json.load(f)
    
    if not results:
        print("没有找到结果文件!")
        return
    
    # 打印对比表格
    print("\n测试集性能对比:")
    print("-" * 80)
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Best Epoch':<12}")
    print("-" * 80)
    
    for model_type in ['resnet', 'lstm', 'cnn_rnn']:
        if model_type in results:
            m = results[model_type]['test_metrics']
            print(f"{model_type.upper():<15} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1']:<12.4f} {results[model_type]['best_epoch']:<12}")
    print("-" * 80)
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['test_metrics']['f1'])
    print(f"\n最佳模型 (按F1): {best_model[0].upper()}")
    print(f"  F1-Score: {best_model[1]['test_metrics']['f1']:.4f}")
    
    # 保存对比结果
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    with open(SAVE_DIR / "comparison_summary.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n[✓] 对比结果已保存: {SAVE_DIR / 'comparison_summary.json'}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='USTC-TFC2016 模型横向对比训练')
    parser.add_argument('--model', choices=['resnet', 'lstm', 'cnn_rnn', 'all'], default='all')
    parser.add_argument('--batch_size', type=int, default=64, help='USTC数据量小，建议用较小batch')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.model != 'all':
        train_model_ustc(args.model, batch_size=args.batch_size, 
                        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)
    else:
        for model_type in ['resnet', 'lstm', 'cnn_rnn']:
            try:
                train_model_ustc(model_type, batch_size=args.batch_size,
                                epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)
            except Exception as e:
                print(f"[✗] {model_type} 训练失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 训练完成后对比结果
        compare_results()