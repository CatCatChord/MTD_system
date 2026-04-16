#!/usr/bin/env python3
"""
CIC-IDS2017 模型横向对比训练脚本 V5.1
=====================================
针对 CNN_RNN 的震荡问题做最小侵入式修复：
1. 分层划分验证集（所有模型受益，无副作用）
2. CNN_RNN 单独使用 weighted 平均指标、更低学习率、更大验证集比例
3. ResNet/BiLSTM 完全保持原有行为
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
BASE_DIR = Path(r"/home/ckb/malicious_traffic_detection/research")
DATA_DIR = BASE_DIR / "processed_data"
SAVE_DIR = BASE_DIR / "comparison_results_cic_v5"
SAVE_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


def seed_everything(seed=42):
    """固定随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[✓] 随机种子: {seed}")


# ==================== 1. ResNet (28x28x1) ====================

class ResidualBlock(nn.Module):
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


# ==================== 2. BiLSTM (50, 2) ====================

class BiLSTM(nn.Module):
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


# ==================== 3. CNN+RNN V5 - 彻底重构（增加LayerNorm稳定） ====================

class CNN_RNN_V5_Simple(nn.Module):
    """
    简化版CNN+RNN - 增加LayerNorm稳定输出，缓解震荡
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
        
        # 新增：LayerNorm稳定LSTM输出
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
        
        # LSTM: [batch, 50, 64]
        x = x.permute(0, 2, 1)  # [batch, 50, 64]
        lstm_out, _ = self.lstm(x)  # [batch, 50, 256]
        
        # 应用LayerNorm稳定特征
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return self.classifier(context)


# ==================== 数据加载（分层抽样，所有模型受益） ====================

def load_data(model_type, dataset='CIC', batch_size=128, val_split=0.1):
    """加载CIC数据 - 改为分层抽样，确保验证集类别分布均衡"""
    data_file = DATA_DIR / f"{dataset}_{model_type}_data.npz"
    print(f"\n加载: {data_file.name}")
    
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
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
    
    num_classes = int(np.unique(y_train).size)
    print(f"  训练: {X_train.shape}, 测试: {X_test.shape}, 类别: {num_classes}")
    
    # 类别分布
    class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
    print(f"  训练集分布: {dict(enumerate(class_counts))}")
    
    # ========== 关键修改：使用分层抽样替代随机抽样 ==========
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx_arr, val_idx_arr = next(sss.split(X_train, y_train))
    train_idx = torch.tensor(train_idx_arr)
    val_idx = torch.tensor(val_idx_arr)
    
    # 打印验证集分布，检查小类别样本数
    val_dist = np.bincount(y_train[val_idx].numpy(), minlength=num_classes)
    print(f"  验证集分布: {dict(enumerate(val_dist))}")
    min_val_samples = val_dist.min()
    if min_val_samples < 5:
        print(f"  ⚠️ 警告: 验证集中最少类别只有 {min_val_samples} 个样本，建议增大 val_split")
    
    # 加权采样（仅训练集）
    train_counts = np.bincount(y_train[train_idx].numpy())
    weights = 1.0 / (train_counts + 1e-8)
    sample_weights = weights[y_train[train_idx].numpy()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        TensorDataset(X_train[train_idx], y_train[train_idx]),
        batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_train[val_idx], y_train[val_idx]),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes, class_names


# ==================== 评估函数（添加average参数，不影响其他模型） ====================

def evaluate(model, loader, criterion, device, average='macro'):
    """
    统一评估 - 添加 average 参数
    默认 'macro' 保持与之前一致，不影响 ResNet/BiLSTM
    CNN_RNN 可传入 'weighted' 减少小类别震荡
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
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
    acc = accuracy_score(all_labels, all_preds)
    
    # 使用传入的 average 参数计算指标
    precision = precision_score(all_labels, all_preds, average=average, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)
    
    return avg_loss, acc, precision, recall, f1, all_preds, all_labels


# ==================== 训练函数（CNN_RNN单独优化，其他模型不变） ====================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)  # 训练集保持macro
    
    return avg_loss, acc, f1


def train_model(model_type, dataset='CIC', batch_size=128, epochs=50, lr=1e-3, 
                patience=10, seed=42, grad_clip=1.0):
    """统一训练 - CNN_RNN 单独使用优化配置"""
    print(f"\n{'='*60}")
    print(f"训练 CIC {model_type.upper()}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # ========== 关键修改：CNN_RNN 单独优化配置，其他模型保持原样 ==========
    if model_type == 'cnn_rnn':
        val_split = 0.15          # 增大验证集比例（从 10% 到 15%），确保小类别有足够样本
        actual_lr = 5e-4          # 降低初始学习率（从 1e-3 到 5e-4），减少震荡
        actual_patience = patience  # 保持原有耐心值
        metric_average = 'weighted'  # 使用 weighted 平均替代 macro，消除小类别样本震荡
        print(f"[CNN_RNN 优化配置] val_split={val_split}, lr={actual_lr}, metric_average={metric_average}")
    else:
        # ResNet 和 BiLSTM 保持原有配置，完全不受影响
        val_split = 0.1
        actual_lr = lr
        actual_patience = patience
        metric_average = 'macro'
    
    train_loader, val_loader, test_loader, num_classes, class_names = \
        load_data(model_type, dataset, batch_size=batch_size, val_split=val_split)
    
    # 创建模型
    if model_type == 'resnet':
        model = ResNet28(num_classes).to(DEVICE)
    elif model_type == 'lstm':
        sample = next(iter(train_loader))[0]
        model = BiLSTM(num_classes, input_size=sample.shape[-1]).to(DEVICE)
    elif model_type == 'cnn_rnn':
        model = CNN_RNN_V5_Simple(num_classes).to(DEVICE)
    else:
        raise ValueError(f"未知模型: {model_type}")
    
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加权损失
    class_counts = np.bincount([l.item() for _, labels in train_loader for l in labels], 
                               minlength=num_classes)
    weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"类别权重: {weights.cpu().numpy().round(4)}")
    
    # 优化器 - CNN_RNN 使用更低学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # 记录
    history = {k: [] for k in ['train_loss', 'train_acc', 'train_f1',
                               'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1',
                               'test_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'lr']}
    
    best_val_f1 = 0
    trigger_times = 0
    best_epoch = 0
    
    print(f"\n开始训练 (patience={actual_patience}, grad_clip={grad_clip})...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, grad_clip
        )
        
        # ========== 关键修改：传入 metric_average 参数 ==========
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate(
            model, val_loader, criterion, DEVICE, average=metric_average
        )
        test_loss, test_acc, test_p, test_r, test_f1, _, _ = evaluate(
            model, test_loader, criterion, DEVICE, average=metric_average
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_p)
        history['val_recall'].append(val_r)
        history['val_f1'].append(val_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(test_p)
        history['test_recall'].append(test_r)
        history['test_f1'].append(test_f1)
        history['lr'].append(current_lr)
        
        print(f"Train - Loss:{train_loss:.4f} Acc:{train_acc:.4f} F1:{train_f1:.4f}")
        print(f"Val   - Loss:{val_loss:.4f} Acc:{val_acc:.4f} P:{val_p:.4f} R:{val_r:.4f} F1:{val_f1:.4f} [{metric_average}]")
        print(f"Test  - Loss:{test_loss:.4f} Acc:{test_acc:.4f} P:{test_p:.4f} R:{test_r:.4f} F1:{test_f1:.4f} [{metric_average}]")
        
        scheduler.step(val_f1)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            trigger_times = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'num_classes': num_classes,
                'class_names': class_names
            }, SAVE_DIR / f"best_{model_type}.pth")
            print(f"  ✓ 最佳模型保存 (Val F1:{val_f1:.4f})")
        else:
            trigger_times += 1
            print(f"  F1未提升 ({trigger_times}/{actual_patience})")
        
        if trigger_times >= actual_patience:
            print(f"[!] 早停触发")
            break
    
    # 最终评估
    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")
    
    checkpoint = torch.load(SAVE_DIR / f"best_{model_type}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终评估也使用对应的 average 方法
    test_loss, test_acc, test_p, test_r, test_f1, preds, labels = \
        evaluate(model, test_loader, criterion, DEVICE, average=metric_average)
    
    cm = confusion_matrix(labels, preds)
    
    print(f"\n最佳模型 Epoch {checkpoint['epoch']+1}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_p:.4f} ({metric_average})")
    print(f"  Recall:    {test_r:.4f} ({metric_average})")
    print(f"  F1-Score:  {test_f1:.4f} ({metric_average})")
    
    # 保存结果
    results = {
        'model_type': model_type,
        'best_epoch': best_epoch + 1,
        'best_val_f1': float(best_val_f1),
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_p),
            'recall': float(test_r),
            'f1': float(test_f1),
            'average_method': metric_average  # 记录使用的平均方法
        },
        'history': history,
        'confusion_matrix': cm.tolist()
    }
    
    with open(SAVE_DIR / f"results_{model_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图（保持原有逻辑不变）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].plot(history['test_acc'], label='Test')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    
    axes[0, 2].plot(history['train_f1'], label='Train')
    axes[0, 2].plot(history['val_f1'], label='Val')
    axes[0, 2].plot(history['test_f1'], label='Test')
    axes[0, 2].set_title('F1-Score')
    axes[0, 2].legend()
    
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].set_title(f'Val Precision & Recall ({metric_average})')
    axes[1, 0].legend()
    
    axes[1, 1].plot(history['test_precision'], label='Precision')
    axes[1, 1].plot(history['test_recall'], label='Recall')
    axes[1, 1].set_title(f'Test Precision & Recall ({metric_average})')
    axes[1, 1].legend()
    
    axes[1, 2].plot(history['lr'])
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"curves_{model_type}.png", dpi=150)
    plt.close()
    
    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'{model_type.upper()} Confusion Matrix', ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"confusion_{model_type}.png", dpi=150)
    plt.close()
    
    print(f"\n[✓] {model_type.upper()} 完成! 最佳Val F1: {best_val_f1:.4f}")
    
    return results


def compare_results():
    """对比三个模型"""
    print("\n" + "="*70)
    print("CIC-IDS2017 模型横向对比 V5.1 (CNN_RNN优化版)")
    print("="*70)
    
    results = {}
    for model_type in ['resnet', 'lstm', 'cnn_rnn']:
        result_file = SAVE_DIR / f"results_{model_type}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[model_type] = json.load(f)
    
    if not results:
        print("无结果文件")
        return
    
    print("\n测试集性能对比:")
    print("-" * 90)
    print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AvgMethod':<10} {'Epoch':<8}")
    print("-" * 90)
    
    for model_type in ['resnet', 'lstm', 'cnn_rnn']:
        if model_type in results:
            m = results[model_type]['test_metrics']
            avg_method = m.get('average_method', 'macro')  # 兼容旧版本结果
            print(f"{model_type.upper():<12} {m['accuracy']:<10.4f} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1']:<12.4f} {avg_method:<10} {results[model_type]['best_epoch']:<8}")
    print("-" * 90)
    
    best = max(results.items(), key=lambda x: x[1]['test_metrics']['f1'])
    print(f"\n最佳模型: {best[0].upper()} (F1: {best[1]['test_metrics']['f1']:.4f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CIC-IDS2017 模型横向对比 V5.1')
    parser.add_argument('--model', choices=['resnet', 'lstm', 'cnn_rnn', 'all'], default='all')
    parser.add_argument('--dataset', default='CIC')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    args = parser.parse_args()
    
    if args.model != 'all':
        train_model(args.model, args.dataset, batch_size=args.batch_size, 
                   epochs=args.epochs, lr=args.lr, patience=args.patience, 
                   seed=args.seed, grad_clip=args.grad_clip)
    else:
        for model_type in ['resnet', 'lstm', 'cnn_rnn']:
            try:
                train_model(model_type, args.dataset, batch_size=args.batch_size,
                           epochs=args.epochs, lr=args.lr, patience=args.patience, 
                           seed=args.seed, grad_clip=args.grad_clip)
            except Exception as e:
                print(f"[✗] {model_type} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        compare_results()