#!/usr/bin/env python3
"""
USTC-TFC2016 模型训练脚本 - 修复版
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEQ_LEN = 200  # 改为 200，避免 RNN 丢失太多序列信息



def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, mode='max', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            self.early_stop = False
            if self.restore_best_weights and model is not None:
                self.best_model_state = model.state_dict()
            return

        improved = score > self.best_score + self.min_delta if self.mode == 'max' else score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.early_stop = False
            if self.restore_best_weights and model is not None:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def apply_best_weights(self, model):
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ==================== 配置 ====================
BASE_DIR = Path(r"/home/ckb/malicious_traffic_detection/research")
DATA_DIR = BASE_DIR / "processed_data"
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

# USTC预处理输出目录（与process_ustc.py一致）
USTC_DATA_DIR = Path("/mnt/data/processed_data_fixed")

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ==================== 数据加载 ====================
def load_ustc_data(model_type):
    """加载USTC预处理数据"""
    candidates = [
        USTC_DATA_DIR / f"USTC_{model_type}_fixed.npz",
        DATA_DIR / f"USTC_{model_type}_fixed.npz",
        DATA_DIR / f"USTC_{model_type}_data.npz",
    ]

    data_file = None
    for p in candidates:
        if p.exists():
            data_file = p
            break

    if data_file is None:
        raise FileNotFoundError(
            f"找不到数据文件: 候选路径为 {candidates} (请先运行 process_ustc.py)"
        )

    print(f"\n加载数据: {data_file}")
    data = np.load(data_file)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # 可能存在 int64 或 float32
    if X_train.dtype == np.int64 and model_type != 'cnn_rnn':
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    if y_train.dtype != np.int64:
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

    num_classes = int(np.unique(y_train).size)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"类别数: {num_classes}")
    print(f"类别分布 - 训练集: {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test, num_classes


def check_data_overlap():
    """检查多个模型文件是否存在重叠样本，避免索引误用或数据混淆"""
    files = {
        m: USTC_DATA_DIR / f"USTC_{m}_fixed.npz"
        for m in ['resnet', 'lstm', 'cnn_rnn']
    }
    valid = {m: p for m, p in files.items() if p.exists()}
    if len(valid) < 2:
        return

    contents = {m: np.load(path)['X_train'] for m, path in valid.items()}
    keys = list(contents.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            k = min(10, contents[a].shape[0], contents[b].shape[0])
            if np.array_equal(contents[a][:k], contents[b][:k]):
                print(f"[警告] {a} 与 {b} 前{k} 个训练样本完全相同，可能存在数据混用")


def prepare_ustc_loaders(X_train, X_test, y_train, y_test, model_type, batch_size=64):
    """准备数据加载器"""
    # 维度调整（与之前相同）
    if model_type == 'resnet':
        # ResNet处理在process_ustc中已做64x64、3通道、[-1,1]归一化
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)
        elif len(X_train.shape) == 2:
            size = int(np.sqrt(X_train.shape[1]))
            if size * size == X_train.shape[1]:
                X_train = X_train.reshape(-1, 1, size, size)
                X_test = X_test.reshape(-1, 1, size, size)
            else:
                target_size = 64 * 64
                if X_train.shape[1] >= target_size:
                    X_train = X_train[:, :target_size].reshape(-1, 1, 64, 64)
                    X_test = X_test[:, :target_size].reshape(-1, 1, 64, 64)
                else:
                    pad_width = ((0, 0), (0, target_size - X_train.shape[1]))
                    X_train = np.pad(X_train, pad_width, mode='constant').reshape(-1, 1, 64, 64)
                    X_test = np.pad(X_test, pad_width, mode='constant').reshape(-1, 1, 64, 64)

    elif model_type in ['lstm', 'cnn_rnn']:
        if len(X_train.shape) == 2:
            seq_len = SEQ_LEN
            if X_train.shape[1] % seq_len == 0:
                feat_dim = X_train.shape[1] // seq_len
                X_train = X_train.reshape(-1, seq_len, feat_dim)
                X_test = X_test.reshape(-1, seq_len, feat_dim)
            else:
                # 常用USTC特征维度为3，如果不能整分，截断/填充到 SEQ_LEN*3
                feat_dim = 3
                total_features = seq_len * feat_dim

                if X_train.shape[1] < total_features:
                    pad_width = total_features - X_train.shape[1]
                    X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
                    X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    X_train = X_train[:, :total_features]
                    X_test = X_test[:, :total_features]

                X_train = X_train.reshape(-1, seq_len, feat_dim)
                X_test = X_test.reshape(-1, seq_len, feat_dim)

            # 长度超过50则截断，过短则补零
            if X_train.shape[1] > seq_len:
                X_train = X_train[:, :seq_len, :]
                X_test = X_test[:, :seq_len, :]
            elif X_train.shape[1] < seq_len:
                pad_len = seq_len - X_train.shape[1]
                X_train = np.pad(X_train, ((0,0), (0,pad_len), (0,0)), mode='constant')
                X_test = np.pad(X_test, ((0,0), (0,pad_len), (0,0)), mode='constant')

        # 标准化
        mean, std = X_train.mean(), X_train.std() + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    
    print(f"调整后数据形状: {X_train.shape}")
    
    # 类别权重处理（解决不平衡）
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), 
                                    num_samples=len(y_train), replacement=True)
    
    if model_type == 'cnn_rnn':
        # cnn_rnn 使用 int64 byte payload 值（0-255）
        train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    else:
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train.shape

# ==================== 模型定义（与之前相同）====================
class TrueResNetUSTC(nn.Module):
    def __init__(self, num_classes=20, in_channels=1):
        super(TrueResNetUSTC, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNetUSTC(nn.Module):
    def __init__(self, num_classes=20, in_channels=3):
        super(ResNetUSTC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)

class LSTMUSTC(nn.Module):
    def __init__(self, num_classes=20, input_size=3, hidden_size=20, num_layers=1, embed_dim=16):
        super(LSTMUSTC, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim

        self.seq_dim = min(3, input_size)
        self.use_stat = (input_size > 3)
        self.stat_dim = max(0, input_size - self.seq_dim)

        self.embedding = nn.Embedding(256, embed_dim, padding_idx=0)
        self.lstm_input_size = self.seq_dim * embed_dim
        self.seq_proj = nn.Linear(self.seq_dim, self.lstm_input_size)

        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.0)

        if self.use_stat:
            self.bn_stats = nn.BatchNorm1d(self.stat_dim)
            self.stat_fc = nn.Sequential(
                nn.Linear(self.stat_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + 32, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )

    def forward(self, x):
        # x: [B, Seq, feat_dim]
        # If input is float and has >3, treat as seq+stat.
        if x.dtype != torch.long:
            x = x.float()

        # 流量字节离散值结构：只要是在 0-255 之间的整型，则使用 Embedding 处理
        seq_feat = x[:, :, :self.seq_dim]
        if x.dtype == torch.long and x.max() <= 255:
            seq_emb = self.embedding(seq_feat)
            seq_emb = seq_emb.view(x.size(0), x.size(1), -1)
        else:
            # 连续数值输入，线性投影到与embedding一致的维度
            seq_emb = self.seq_proj(seq_feat)

        lstm_out, _ = self.lstm(seq_emb)
        lstm_final = lstm_out[:, -1, :]

        if self.use_stat and x.size(2) > 3:
            stat_feat = x[:, 0, 3:]
            stat_norm = self.bn_stats(stat_feat)
            stat_emb = self.stat_fc(stat_norm)
            combined = torch.cat([lstm_final, stat_emb], dim=1)
            return self.classifier(combined)

        return self.classifier(lstm_final)

class CNNRNN_USTC(nn.Module):
    def __init__(self, num_classes=2, input_channels=256, embed_dim=16):
        super(CNNRNN_USTC, self).__init__()
        # process_ustc 中CNN-RNN输入已为 int (0-255), 使用Embedding更符合脚本设计
        self.embedding = nn.Embedding(256, embed_dim, padding_idx=0)
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(25)
        )
        self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len=50, packet_bytes=256)
        if x.dtype != torch.long:
            x = x.long()

        # 保存原始输入索引，用于 padding mask (0 为 padding)
        x_indices = x

        # Embedding: (B, 50, 256, embed_dim)
        x = self.embedding(x_indices)

        # mask padding 0 的字节，避免Embedding将padding也纳入均值
        # mask: (B, 50, 256, 1)
        mask = (x_indices != 0).unsqueeze(-1).float()
        x = x * mask

        # 先对字节维度做平均池化 -> (B, 50, embed_dim)
        # mask_sum: (B, 50, 1)
        mask_sum = mask.sum(dim=2).clamp(min=1.0)
        x = x.sum(dim=2) / mask_sum

        # 变换为如下结构，用CNN提取局部特征 (B, embed_dim, 50)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # LSTM输入 (B, 25, 64)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])

# ==================== 训练流程 ====================
def train_epoch_ustc(model, loader, optimizer, criterion, device, scaler=None, grad_clip=2.0, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        if scaler is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'} )

    return total_loss / len(loader), correct / total

def evaluate_ustc(model, loader, device):
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
    f1 = f1_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    
    return acc, f1, recall, all_preds, all_targets

def train_ustc_model(model_type, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                     optimizer_type='adamw', scheduler_type='cosine', weight_decay=1e-4,
                     patience=10, min_delta=0.005):
    """主训练函数 - 修复版"""
    print(f"\n{'='*60}")
    print(f"USTC-TFC2016 训练 {model_type.upper()}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(42)
    
    X_train, X_test, y_train, y_test, num_classes = load_ustc_data(model_type)
    check_data_overlap()
    train_loader, test_loader, input_shape = prepare_ustc_loaders(
        X_train, X_test, y_train, y_test, model_type, batch_size
    )
    
    # 创建模型
    if model_type == 'resnet':
        in_channels = input_shape[1] if len(input_shape) == 4 else 1
        model = TrueResNetUSTC(num_classes, in_channels=in_channels).to(DEVICE)
    elif model_type == 'lstm':
        input_size = input_shape[2] if len(input_shape) == 3 else 3
        model = LSTMUSTC(num_classes, input_size=input_size).to(DEVICE)
    elif model_type == 'cnn_rnn':
        input_channels = input_shape[2] if len(input_shape) == 3 else 256
        model = CNNRNN_USTC(num_classes, input_channels).to(DEVICE)
    
    print(f"\n模型: {model_type}, 输入: {input_shape}")
    print(model)

    # 类别不平衡训练损失加权 + Label Smoothing
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == 'cosine_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
            last_epoch=-1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max', restore_best_weights=True)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_f1': []}
    best_f1 = 0.0
    current_lr = lr
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch_ustc(
            model, train_loader, optimizer, criterion, DEVICE, scaler,
            scheduler=scheduler if scheduler_type == 'onecycle' else None
        )
        test_acc, test_f1, test_recall, preds, targets = evaluate_ustc(model, test_loader, DEVICE)
        
        # 学习率调度（不同策略有不同触发方式）
        if scheduler_type == 'cosine':
            scheduler.step()
        elif scheduler_type == 'cosine_restart':
            scheduler.step()
        elif scheduler_type == 'plateau':
            scheduler.step(train_loss)
        # onecycle 在每个batch内部step，epoch级别不额外调用
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f}")

        # 保存最佳模型文件至少应按 F1 选取，针对非对称类别重要
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_f1': test_f1,
                'num_classes': num_classes,
                'model_type': model_type,
            }, SAVE_DIR / f"USTC_{model_type}_best.pth")
            print(f"  ✓ 保存最佳模型 (F1: {test_f1:.4f})")

        early_stopping(test_f1, model)
        if early_stopping.early_stop:
            print(f"[!] 早停触发 (patience={early_stopping.patience})，提前结束训练")
            break
    
    # 早停后回滚到最优权重（如果可用）
    early_stopping.apply_best_weights(model)

    # 保存最终模型
    torch.save(model.state_dict(), SAVE_DIR / f"USTC_{model_type}_final.pth")
    
    # 保存历史
    with open(SAVE_DIR / f"USTC_{model_type}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘图
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'])
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['test_f1'])
    plt.title('Test F1-Score')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"USTC_{model_type}_curves.png", dpi=150)
    plt.close()
    
    # 最终报告
    print(f"\n{'='*60}")
    print("最终评估")
    print(f"最佳测试F1: {best_f1:.4f}")
    print(f"最终F1分数: {test_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(targets, preds, digits=4))
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='USTC-TFC2016 Traffic Classification')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['resnet', 'lstm', 'cnn_rnn', 'all'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'cosine_restart', 'onecycle', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.005)

    args = parser.parse_args()
    
    model_list = ['resnet', 'lstm', 'cnn_rnn'] if args.model == 'all' else [args.model]
    
    for model_type in model_list:
        try:
            train_ustc_model(model_type,
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             lr=args.lr,
                             optimizer_type=args.optimizer,
                             scheduler_type=args.scheduler,
                             weight_decay=args.weight_decay,
                             patience=args.patience,
                             min_delta=args.min_delta)
        except Exception as e:
            print(f"\n[✗] {model_type} 训练失败: {e}")
            import traceback
            traceback.print_exc()