#!/usr/bin/env python3
"""
消融实验脚本 (Ablation Study)
=============================
基于现有小型数据集 (X_lstm.npy / X_resnet.npy / y_labels.npy) 快速验证关键组件的作用。

支持三种消融：
1. BiLSTM -Attention : 去掉 Self-Attention，仅取最后时刻隐藏状态
2. BiLSTM -Bidirectional : 单向 LSTM
3. ResNet -Skip : 去掉残差连接

用法:
    python run_ablation.py --model lstm
    python run_ablation.py --model resnet
    python run_ablation.py --model all
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置 ====================
DATA_DIR = Path(__file__).parent / "processed_data"
SAVE_DIR = Path(__file__).parent / "ablation_results"
SAVE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 消融模型定义 ====================

class BiLSTMCIC_Full(nn.Module):
    """完整版: BiLSTM + Attention (对应 backend/app/models.py)"""
    def __init__(self, num_classes=2, input_size=2, hidden=128, layers=2, dropout=0.3):
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
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context)


class BiLSTMCIC_NoAttn(nn.Module):
    """消融1: 去掉 Attention，只取最后时刻隐藏状态"""
    def __init__(self, num_classes=2, input_size=2, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 直接取最后时刻
        final = lstm_out[:, -1, :]
        return self.classifier(final)


class BiLSTMCIC_NoBidirectional(nn.Module):
    """消融2: 单向 LSTM"""
    def __init__(self, num_classes=2, input_size=2, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                           batch_first=True, bidirectional=False,
                           dropout=dropout if layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        return self.classifier(final)


class ResNetCIC_Full(nn.Module):
    """完整版: ResNet28 with Skip Connection"""
    def __init__(self, num_classes=2, dropout=0.3):
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
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
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


class ResNetCIC_NoSkip(nn.Module):
    """消融3: 普通 CNN，去掉残差连接"""
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """残差块 (供完整版 ResNet 使用)"""
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


# ==================== 数据加载 ====================

def load_npy_data(model_type='lstm'):
    """加载本地 .npy 小样本数据 (2484 条, 二分类)"""
    if model_type == 'lstm':
        X = np.load(DATA_DIR / "X_lstm.npy")
    elif model_type == 'resnet':
        X = np.load(DATA_DIR / "X_resnet.npy")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    y = np.load(DATA_DIR / "y_labels.npy")

    # 划分训练/测试集 (8:2)
    n_total = len(y)
    indices = np.random.permutation(n_total)
    split = int(n_total * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"[数据] {model_type}: 训练集 {len(train_ds)}, 测试集 {len(test_ds)}, 类别 {np.unique(y)}")
    return train_loader, test_loader


# ==================== 训练与评估 ====================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1, all_preds, all_targets


def run_ablation(name, model, train_loader, test_loader, epochs=20, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"开始消融: {name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)

    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_acc, prec, rec, f1, preds, targets = evaluate(model, test_loader, DEVICE)
        scheduler.step(1.0 - test_acc)
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'test_acc': test_acc, 'precision': prec, 'recall': rec, 'f1': f1
        })
        if test_acc > best_acc:
            best_acc = test_acc
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | TrainAcc: {train_acc:.4f} | TestAcc: {test_acc:.4f} | F1: {f1:.4f}")

    print(f"[结果] {name} 最佳测试准确率: {best_acc:.4f}")
    return history, best_acc, preds, targets


def plot_ablation_curves(results, save_path):
    """绘制消融实验对比曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, (history, _, _, _) in results.items():
        epochs = [h['epoch'] for h in history]
        test_acc = [h['test_acc'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        axes[0].plot(epochs, test_acc, marker='o', label=name, markersize=3)
        axes[1].plot(epochs, train_loss, label=name)

    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title('Train Loss Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 消融对比曲线: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 混淆矩阵: {save_path}")


# ==================== 主入口 ====================

ABLATION_CONFIG = {
    'lstm': [
        ("BiLSTM-Full", BiLSTMCIC_Full),
        ("BiLSTM-NoAttn", BiLSTMCIC_NoAttn),
        ("BiLSTM-NoBi", BiLSTMCIC_NoBidirectional),
    ],
    'resnet': [
        ("ResNet-Full", ResNetCIC_Full),
        ("ResNet-NoSkip", ResNetCIC_NoSkip),
    ]
}


def main():
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'resnet', 'all'])
    parser.add_argument('--epochs', type=int, default=20, help='每个变体训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)
    models_to_run = ['lstm', 'resnet'] if args.model == 'all' else [args.model]

    summary = []

    for model_type in models_to_run:
        train_loader, test_loader = load_npy_data(model_type)
        results = {}

        for name, ModelCls in ABLATION_CONFIG[model_type]:
            model = ModelCls(num_classes=2).to(DEVICE)
            history, best_acc, preds, targets = run_ablation(
                name, model, train_loader, test_loader, epochs=args.epochs, lr=args.lr
            )
            results[name] = (history, best_acc, preds, targets)
            summary.append({
                'base_model': model_type,
                'variant': name,
                'best_test_acc': round(best_acc, 4)
            })

            # 保存每个变体的混淆矩阵
            plot_confusion_matrix(
                targets, preds, classes=['Normal', 'Malicious'],
                title=f"{name} Confusion Matrix",
                save_path=SAVE_DIR / f"cm_{name.replace(' ', '_')}.png"
            )

        # 绘制对比曲线
        plot_ablation_curves(results, SAVE_DIR / f"ablation_{model_type}.png")

        # 保存原始结果 JSON
        with open(SAVE_DIR / f"ablation_{model_type}_results.json", 'w', encoding='utf-8') as f:
            json.dump({k: {'best_acc': v[1], 'history': v[0]} for k, v in results.items()}, f, indent=2)

    # 打印总结表格
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'基础模型':<12} {'变体':<25} {'最佳测试准确率':<15}")
    print("-"*60)
    for row in summary:
        print(f"{row['base_model']:<12} {row['variant']:<25} {row['best_test_acc']:<15}")
    print("="*60)

    # 保存汇总
    with open(SAVE_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
