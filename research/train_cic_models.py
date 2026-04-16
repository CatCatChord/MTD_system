#!/usr/bin/env python3
"""
模型训练脚本 - 支持ResNet/LSTM/CNN+RNN
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# 配置
BASE_DIR = Path(r"/home/ckb/malicious_traffic_detection/research")
DATA_DIR = BASE_DIR / "processed_data"
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

# 默认参数
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


def set_seed(seed=42):
    """设置随机种子，方便复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(model_type, dataset='CIC', batch_size=DEFAULT_BATCH_SIZE, num_workers=4, val_split=0.0):
    """加载数据并返回 loaders"""
    """加载数据"""
    data_file = DATA_DIR / f"{dataset}_{model_type}_data.npz"
    data = np.load(data_file)

    X_train = torch.FloatTensor(data['X_train'])
    X_test = torch.FloatTensor(data['X_test'])
    y_train = torch.LongTensor(data['y_train'])
    y_test = torch.LongTensor(data['y_test'])

    num_classes = int(np.unique(data['y_train']).size)

    # 可选验证集拆分
    val_loader = None
    if 0.0 < val_split < 1.0:
        val_n = int(len(X_train) * val_split)
        idx = torch.randperm(len(X_train))
        train_idx, val_idx = idx[val_n:], idx[:val_n]
        train_set = TensorDataset(X_train[train_idx], y_train[train_idx])
        val_set = TensorDataset(X_train[val_idx], y_train[val_idx])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}, 分类数: {num_classes}")
    if val_loader is not None:
        print(f"验证集: {val_n} 条样本")

    return train_loader, test_loader, val_loader, num_classes


# ==================== 模型定义 ====================

class ResNetTraffic(nn.Module):
    """ResNet for 28x28 traffic images"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LSTMTraffic(nn.Module):
    """LSTM for sequence (batch, 50, 2)"""
    def __init__(self, num_classes, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 50, 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后时刻的隐藏状态
        out = self.classifier(lstm_out[:, -1, :])
        return out


class CNNRNNTraffic(nn.Module):
    """CNN+RNN hybrid for (batch, 50, 256)"""
    def __init__(self, num_classes):
        super().__init__()
        # 1D-CNN处理每个包的空间特征
        self.cnn = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(25)  # 输出25维
        )
        # LSTM处理时序
        self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, 50, 256) -> 需要转置为 (batch, 256, 50) 给CNN1d
        x = x.permute(0, 2, 1)  # (batch, 256, 50)
        x = self.cnn(x)         # (batch, 64, 25)
        x = x.permute(0, 2, 1)  # (batch, 25, 64) 给LSTM
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out[:, -1, :])
        return out


# ==================== 训练流程 ====================

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None and device.type == 'cuda')):
            output = model(data)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    acc = correct / total
    return acc, all_preds, all_targets


class EarlyStopping:
    """早停机制：若验证集指标不再提升，则停止训练"""
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(model_type, dataset='CIC', batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, val_split=0.1, patience=8):
    """主训练函数"""
    print(f"\n{'='*60}")
    print(f"训练 {model_type.upper()} - {dataset}")
    print(f"{'='*60}")
    
    # 设置种子
    set_seed()

    # 加载数据
    train_loader, test_loader, val_loader, num_classes = load_data(model_type, dataset, batch_size=batch_size, val_split=val_split)
    
    # 创建模型
    if model_type == 'resnet':
        model = ResNetTraffic(num_classes).to(DEVICE)
    elif model_type == 'lstm':
        model = LSTMTraffic(num_classes).to(DEVICE)
    elif model_type == 'cnn_rnn':
        model = CNNRNNTraffic(num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    early_stopping = EarlyStopping(patience=patience)

    # 训练记录
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)

        # 验证
        val_acc = None
        if val_loader is not None:
            val_acc, _, _ = evaluate(model, val_loader, DEVICE)

        # 测试
        test_acc, _, _ = evaluate(model, test_loader, DEVICE)
        
        monitor_metric = val_acc if val_acc is not None else test_acc
        scheduler.step(1.0 - monitor_metric)
        early_stopping(monitor_metric)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc if val_acc is not None else 0.0)
        history['test_acc'].append(test_acc)
        
        if val_acc is not None:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # 保存最佳模型
        if monitor_metric > best_acc:
            best_acc = monitor_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'num_classes': num_classes,
                'model_type': model_type
            }, SAVE_DIR / f"{dataset}_{model_type}_best.pth")
            print(f"  ✓ 保存最佳模型 (Acc: {monitor_metric:.4f})")

        if early_stopping.should_stop:
            print(f"[!] 早停触发，连续 {patience} 轮无提升，提前结束训练")
            break

    # 保存最终模型和训练历史
    torch.save(model.state_dict(), SAVE_DIR / f"{dataset}_{model_type}_final.pth")
    
    with open(SAVE_DIR / f"{dataset}_{model_type}_history.json", 'w') as f:
        json.dump(history, f)
    
    # 绘制曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(SAVE_DIR / f"{dataset}_{model_type}_curves.png")
    plt.close()
    
    print(f"\n[✓] 训练完成！最佳测试准确率: {best_acc:.4f}")
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练恶意流量分类模型')
    parser.add_argument('--model', choices=['resnet', 'lstm', 'cnn_rnn', 'all'], default='all')
    parser.add_argument('--dataset', default='CIC')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--val_split', type=float, default=0.1, help='训练集拆分比例用于验证集，0表示不拆分')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model != 'all':
        train_model(args.model, args.dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, val_split=args.val_split, patience=args.patience)
    else:
        for model_type in ['resnet', 'lstm', 'cnn_rnn']:
            try:
                train_model(model_type, args.dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, val_split=args.val_split, patience=args.patience)
            except Exception as e:
                print(f"[✗] {model_type} 训练失败: {e}")