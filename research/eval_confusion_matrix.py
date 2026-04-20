#!/usr/bin/env python3
"""
混淆矩阵与细粒度错误分析脚本
================================
加载现有模型权重，在测试集上生成混淆矩阵热力图。

支持两种模式:
1. 完整 CIC npz 模式: 若 CIC_resnet_data.npz 可用, 绘制 15-class 多分类混淆矩阵
2. 备用 .npy 模式: 使用 X_lstm.npy / X_resnet.npy / y_labels.npy 绘制二分类混淆矩阵

用法:
    python eval_confusion_matrix.py --dataset cic --model lstm
    python eval_confusion_matrix.py --dataset cic --model all
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置 ====================
RESEARCH_DIR = Path(__file__).parent
DATA_DIR = RESEARCH_DIR / "processed_data"
SAVE_DIR = RESEARCH_DIR / "confusion_matrix_results"
SAVE_DIR.mkdir(exist_ok=True)

# 权重搜索路径 (优先 backend/deployed_models, 其次 research/saved_models)
DEPLOYED_CIC = RESEARCH_DIR.parent / "backend" / "deployed_models" / "cicmodel"
DEPLOYED_USTC = RESEARCH_DIR.parent / "backend" / "deployed_models" / "ustcmodel"
SAVED_CIC = RESEARCH_DIR / "saved_models" / "cicmodel"
SAVED_USTC = RESEARCH_DIR / "saved_models" / "ustcmodel"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ==================== 模型定义 (与 backend/app/models.py 及 train_cic_models.py 对齐) ====================

class ResidualBlock(nn.Module):
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
    """USTC ResNet [3,64,64]"""
    def __init__(self, num_classes=20, layers=[2,2,2,2], dropout=0.3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

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
    """USTC BiLSTM + Attention [50,15]"""
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


class CNNRNNTrafficUSTC(nn.Module):
    """USTC CNN+RNN [50,256] int64"""
    def __init__(self, num_classes=20, vocab_size=256, embed_dim=64, seq_len=50, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
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
        if x.dtype != torch.long:
            x = x.long()
        x = self.embedding(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context)
        return out


class ResNet28(nn.Module):
    """CIC ResNet [1,28,28]"""
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


class BiLSTMCIC(nn.Module):
    """CIC BiLSTM + Attention [50,2]"""
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
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context)


class CNNRNNCIC(nn.Module):
    """CIC CNN+RNN [50,256] float"""
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
        x = x.permute(0, 2, 1)
        x = self.cnn_down(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context)


# ==================== 权重加载 ====================

def find_weight_file(dataset, model_name):
    """搜索权重文件路径"""
    candidates = []
    if dataset == 'cic':
        candidates = [
            DEPLOYED_CIC / f"best_{model_name}.pth",
            SAVED_CIC / f"best_{model_name}.pth",
            SAVED_CIC / f"CIC_{model_name}_best.pth",
        ]
    else:
        candidates = [
            DEPLOYED_USTC / f"best_{model_name}.pth",
            SAVED_USTC / f"best_{model_name}.pth",
            SAVED_USTC / f"USTC_{model_name}_best.pth",
        ]

    for p in candidates:
        if p.exists():
            return p
    return None


def load_model_weights(model, weight_path, strict=True):
    """加载权重，兼容多种保存格式"""
    print(f"  加载权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint.get('num_classes', None)
    else:
        state_dict = checkpoint
        num_classes = None

    # 去除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=strict)
    return model, num_classes


# ==================== 数据加载 ====================

def load_cic_data(model_name):
    """尝试加载完整 CIC npz，若损坏则回退到 .npy"""
    npz_path = DATA_DIR / f"CIC_{model_name}_data.npz"

    # 尝试 npz
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            X_test = data['X_test']
            y_test = data['y_test']
            classes = data.get('classes', None)
            if classes is not None:
                # 安全处理可能包含特殊字符的类别名
                class_names = []
                for c in classes:
                    try:
                        s = str(c)
                        # 替换 Windows 终端可能无法打印的字符
                        s = s.encode('utf-8', errors='replace').decode('utf-8')
                        class_names.append(s)
                    except Exception:
                        class_names.append(str(c))
            else:
                class_names = [str(i) for i in np.unique(y_test)]
            print(f"[数据] 成功加载 {npz_path} | 测试集: {X_test.shape} | 类别: {len(class_names)}")
            return X_test, y_test, class_names
        except Exception as e:
            print(f"[警告] {npz_path} 读取失败 ({e})，回退到 .npy 数据")

    # 回退到 .npy (二分类小样本)
    print("[数据] 使用备用 .npy 数据 (二分类)")
    if model_name == 'lstm':
        X = np.load(DATA_DIR / "X_lstm.npy")
    elif model_name == 'resnet':
        X = np.load(DATA_DIR / "X_resnet.npy")
    else:
        # cnn_rnn 没有对应的 .npy，尝试 lstm 或 resnet 的形状都不对
        raise FileNotFoundError(f"找不到 {model_name} 的测试数据。请先运行数据预处理或修复 CIC npz 文件。")

    y = np.load(DATA_DIR / "y_labels.npy")
    # 按 8:2 取测试集 (固定种子保证可复现)
    n_total = len(y)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total)
    split = int(n_total * 0.8)
    test_idx = indices[split:]
    return X[test_idx], y[test_idx], ["Normal", "Malicious"]


def load_ustc_data(model_name):
    """尝试加载 USTC npz 数据"""
    candidates = [
        DATA_DIR / f"USTC_{model_name}_data.npz",
        DATA_DIR / f"USTC_{model_name}_fixed.npz",
        Path("/mnt/data/processed_data_fixed") / f"USTC_{model_name}_fixed.npz",
    ]
    for p in candidates:
        if p.exists():
            data = np.load(p)
            X_test = data['X_test']
            y_test = data['y_test']
            print(f"[数据] 成功加载 {p} | 测试集: {X_test.shape}")
            # USTC 在论文中虽然是20类，但现有预处理为二分类
            classes = ["Normal", "Malicious"]
            return X_test, y_test, classes
    raise FileNotFoundError(f"找不到 USTC 测试数据。候选路径: {candidates}")


def prepare_loader(X, y, model_name, batch_size=128):
    """构造 DataLoader"""
    if model_name == 'cnn_rnn':
        # CIC cnn_rnn 为 float16/255，转为 float32
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    else:
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ==================== 评估与绘图 ====================

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # 按行归一化
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(max(6, len(class_names)*0.5), max(5, len(class_names)*0.4)))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='YlGnBu',
                xticklabels=class_names, yticklabels=class_names, square=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[保存] 混淆矩阵图: {save_path}")


def print_error_analysis(y_true, y_pred, class_names):
    """打印细粒度错误分析"""
    cm = confusion_matrix(y_true, y_pred)
    print("\n[错误分析] 最易混淆的类别对 (Top-5):")
    errors = []
    safe_class_names = []
    for c in class_names:
        try:
            safe_class_names.append(str(c).encode('utf-8', errors='replace').decode('utf-8'))
        except Exception:
            safe_class_names.append(str(c))
    for i in range(len(safe_class_names)):
        for j in range(len(safe_class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], safe_class_names[i], safe_class_names[j]))
    errors.sort(reverse=True)
    for cnt, true_cls, pred_cls in errors[:5]:
        try:
            print(f"  {true_cls} -> {pred_cls}: {cnt} 次")
        except UnicodeEncodeError:
            print(f"  [类别名包含特殊字符] 混淆次数: {cnt} 次")


# ==================== 主入口 ====================

MODEL_CONFIG = {
    'cic': {
        'resnet': (ResNet28, 2),
        'lstm': (BiLSTMCIC, 2),
        'cnn_rnn': (CNNRNNCIC, 2),
    },
    'ustc': {
        'resnet': (ResNetTraffic64, 15),  # 理论上应为20或2，取决于预处理
        'lstm': (BiLSTMTrafficUSTC, 15),
        'cnn_rnn': (CNNRNNTrafficUSTC, 15),
    }
}


def run_single(dataset, model_name):
    print(f"\n{'='*60}")
    print(f"评估 {dataset.upper()} - {model_name.upper()}")
    print(f"{'='*60}")

    # 查找权重
    weight_path = find_weight_file(dataset, model_name)
    if weight_path is None:
        print(f"[跳过] 找不到 {dataset} {model_name} 的权重文件")
        return None

    # 加载数据
    try:
        if dataset == 'cic':
            X_test, y_test, class_names = load_cic_data(model_name)
        else:
            X_test, y_test, class_names = load_ustc_data(model_name)
    except Exception as e:
        print(f"[跳过] 数据加载失败: {e}")
        return None

    # 构建模型
    ModelCls, default_classes = MODEL_CONFIG[dataset][model_name]
    num_classes = len(np.unique(y_test))
    model = ModelCls(num_classes=num_classes).to(DEVICE)

    # 加载权重 (若类别数不匹配则放宽 strict)
    strict = (num_classes == default_classes)
    try:
        model, _ = load_model_weights(model, weight_path, strict=strict)
    except Exception as e:
        print(f"[跳过] 权重加载失败: {e}")
        return None

    # 评估
    loader = prepare_loader(X_test, y_test, model_name)
    acc, prec, rec, f1, preds, targets = evaluate_model(model, loader, DEVICE)
    base_name = f"{dataset}_{model_name}"

    print(f"[指标] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    # 分类报告
    print("\n分类报告:")
    try:
        report = classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0)
        print(report)
    except UnicodeEncodeError:
        # Windows 终端编码问题，保存到文件并打印简化版
        report_path = SAVE_DIR / f"report_{base_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))
        print(f"[注意] 终端编码不支持完整报告，已保存至: {report_path}")
        # 打印简化版指标
        from sklearn.metrics import accuracy_score, f1_score
        print(f"  Accuracy: {accuracy_score(targets, preds):.4f}")
        print(f"  Macro F1: {f1_score(targets, preds, average='macro', zero_division=0):.4f}")

    # 绘制混淆矩阵 (原始计数 + 归一化)
    plot_confusion_matrix(targets, preds, class_names,
                          title=f"{base_name.upper()} Confusion Matrix",
                          save_path=SAVE_DIR / f"cm_{base_name}.png",
                          normalize=False)
    plot_confusion_matrix(targets, preds, class_names,
                          title=f"{base_name.upper()} Normalized CM",
                          save_path=SAVE_DIR / f"cm_{base_name}_norm.png",
                          normalize=True)

    # 错误分析
    print_error_analysis(targets, preds, class_names)

    return {
        'dataset': dataset,
        'model': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'num_classes': num_classes,
        'class_names': class_names,
    }


def main():
    parser = argparse.ArgumentParser(description='混淆矩阵与错误分析')
    parser.add_argument('--dataset', type=str, default='cic', choices=['cic', 'ustc', 'all'])
    parser.add_argument('--model', type=str, default='all', choices=['resnet', 'lstm', 'cnn_rnn', 'all'])
    args = parser.parse_args()

    datasets = ['cic', 'ustc'] if args.dataset == 'all' else [args.dataset]
    models = ['resnet', 'lstm', 'cnn_rnn'] if args.model == 'all' else [args.model]

    summary = []
    for ds in datasets:
        for m in models:
            result = run_single(ds, m)
            if result:
                summary.append(result)

    # 保存汇总
    with open(SAVE_DIR / "evaluation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("评估完成，结果保存在:", SAVE_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
