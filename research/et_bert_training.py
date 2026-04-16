"""
ET-BERT 训练代码 - USTC-TFC2016数据集
基于官方实现: https://github.com/linwhitehat/ET-BERT
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm
import pickle
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== 1. 数据预处理 ====================

class USTCTrafficPreprocessor:
    """
    USTC-TFC2016数据集预处理器
    按照ET-BERT论文要求处理数据包
    """
    
    def __init__(self, max_length=256, packet_per_flow=5):
        """
        Args:
            max_length: 每个数据包的最大字节数（token数）
            packet_per_flow: 每个flow选取的数据包数量
        """
        self.max_length = max_length
        self.packet_per_flow = packet_per_flow
        
    def clean_packet(self, packet_bytes):
        """
        清洗数据包：去除前36字节（Ethernet header + IP + port）
        保留payload部分用于tokenization
        """
        if len(packet_bytes) < 40:  # 丢弃过小的包
            return None
        
        # 去除前36字节（Ethernet: 14, IP: 20, TCP/UDP header部分）
        payload = packet_bytes[36:]
        
        # 如果payload为空或太短，返回None
        if len(payload) < 4:
            return None
            
        return payload
    
    def packet_to_tokens(self, packet_bytes):
        """
        将数据包转换为token序列（每个字节作为一个token）
        ET-BERT使用字节级tokenization
        """
        cleaned = self.clean_packet(packet_bytes)
        if cleaned is None:
            return None
        
        # 截断或填充到固定长度
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]
        else:
            cleaned = cleaned + b'\x00' * (self.max_length - len(cleaned))
        
        # 转换为整数列表（0-255）
        tokens = list(cleaned)
        return tokens
    
    def process_pcap(self, pcap_path, label):
        """
        处理单个pcap文件
        Args:
            pcap_path: pcap文件路径
            label: 类别标签
        Returns:
            samples: 列表，每个元素是一个flow的token序列
        """
        packets = rdpcap(pcap_path)
        
        # 按五元组分组（flow）
        flows = {}
        for pkt in packets:
            if IP in pkt and (TCP in pkt or UDP in pkt):
                # 提取五元组
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                if TCP in pkt:
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                    proto = 'TCP'
                else:
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport
                    proto = 'UDP'
                
                flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
                
                if flow_key not in flows:
                    flows[flow_key] = []
                flows[flow_key].append(bytes(pkt))
        
        # 处理每个flow
        samples = []
        for flow_key, flow_packets in flows.items():
            # 取前packet_per_flow个包
            selected_packets = flow_packets[:self.packet_per_flow]
            
            # 将每个包转换为tokens
            flow_tokens = []
            for pkt_bytes in selected_packets:
                tokens = self.packet_to_tokens(pkt_bytes)
                if tokens is not None:
                    flow_tokens.extend(tokens)
            
            # 确保长度一致（packet_per_flow * max_length）
            target_length = self.packet_per_flow * self.max_length
            if len(flow_tokens) >= target_length * 0.5:  # 至少50%的数据有效
                if len(flow_tokens) > target_length:
                    flow_tokens = flow_tokens[:target_length]
                else:
                    flow_tokens = flow_tokens + [0] * (target_length - len(flow_tokens))
                
                samples.append({
                    'tokens': flow_tokens,
                    'label': label,
                    'flow_id': flow_key
                })
        
        return samples
    
    def process_dataset(self, data_dir, class_names):
        """
        处理整个数据集
        Args:
            data_dir: USTC数据集根目录（含 Malware/Benign 子目录）
            class_names: 类别名称列表
        Returns:
            all_samples: 所有样本列表
        """
        all_samples = []
        malware_dir = os.path.join(data_dir, 'Malware')
        benign_dir = os.path.join(data_dir, 'Benign')

        for label, class_name in enumerate(class_names):
            pcap_paths = []

            # 先找 Malware 下的单文件命名
            malware_file = os.path.join(malware_dir, f"{class_name}.pcap")
            if os.path.isfile(malware_file):
                pcap_paths.append(malware_file)

            # 再找 Benign 下的直文件或子目录
            benign_file = os.path.join(benign_dir, f"{class_name}.pcap")
            if os.path.isfile(benign_file):
                pcap_paths.append(benign_file)
            else:
                benign_subdir = os.path.join(benign_dir, class_name)
                if os.path.isdir(benign_subdir):
                    for f in os.listdir(benign_subdir):
                        if f.endswith('.pcap'):
                            pcap_paths.append(os.path.join(benign_subdir, f))

            # 兼容旧版目录结构（直接在 data_dir 下的类别目录）
            if len(pcap_paths) == 0:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for f in os.listdir(class_dir):
                        if f.endswith('.pcap'):
                            pcap_paths.append(os.path.join(class_dir, f))

            if len(pcap_paths) == 0:
                print(f"Warning: no pcap files found for class {class_name}")
                continue

            print(f"Processing {class_name} ({len(pcap_paths)} files)...")
            for pcap_path in tqdm(pcap_paths):
                try:
                    samples = self.process_pcap(pcap_path, label)
                    all_samples.extend(samples)
                except Exception as e:
                    print(f"Error processing {pcap_path}: {e}")

        return all_samples


# ==================== 2. PyTorch Dataset ====================

class ET_BERT_Dataset(Dataset):
    """ET-BERT数据集"""
    
    def __init__(self, samples, max_length=1280):
        """
        Args:
            samples: 预处理后的样本列表
            max_length: 最大序列长度（BERT的max_position_embeddings）
        """
        self.samples = samples
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        tokens = sample['tokens']
        label = sample['label']
        
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # 创建attention mask
        attention_mask = [1 if token != 0 else 0 for token in tokens]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==================== 3. ET-BERT模型定义 ====================

class ET_BERT_Classifier(nn.Module):
    """
    ET-BERT分类器
    基于BERT架构，使用字节级tokenization
    """
    
    def __init__(self, num_classes=20, vocab_size=256, hidden_size=256, 
                 num_hidden_layers=6, num_attention_heads=8, max_position_embeddings=1280):
        """
        Args:
            num_classes: 分类类别数（USTC-TFC2016有20类）
            vocab_size: 词汇表大小（字节级：256）
            hidden_size: 隐藏层维度
            num_hidden_layers: Transformer层数
            num_attention_heads: 注意力头数
            max_position_embeddings: 最大序列长度
        """
        super().__init__()
        
        # BERT配置
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_position_embeddings,
            num_labels=num_classes,
            type_vocab_size=1,  # 不需要segment embedding
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # BERT模型
        self.bert = BertForSequenceClassification(self.config)
        
        # 已指定num_labels，BertForSequenceClassification初始化时会自动创建分类头
        # 为了确保一致，仍显式设置 classifier
        self.bert.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs


# ==================== 4. 训练函数 ====================

class ET_BERT_Trainer:
    """ET-BERT训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=2e-5, warmup_steps=1000, weight_decay=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度
        total_steps = len(train_loader) * 10  # 假设10个epoch
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 损失函数（处理类别不平衡）
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1, all_labels, all_preds
    
    def train(self, epochs=10, save_dir='./et_bert_checkpoints'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_f1 = 0
        history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                   'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = self.evaluate()
            
            # 记录
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': best_f1,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"✓ Saved best model (F1: {best_f1:.4f})")
        
        return history


# ==================== 5. 主函数 ====================

def main():
    """主函数"""
    # 配置
    DATA_DIR = '/mnt/data/ckb_datasets/USTC-TFC2016'  # USTC数据集路径
    CLASS_NAMES = [
        'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 
        'Tinba', 'Virut', 'Zeus',  # 恶意流量（10类）
        'BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 
        'Skype', 'SMB', 'Weibo', 'WorldOfWarcraft'  # 正常流量（10类）
    ]
    
    # 超参数
    BATCH_SIZE = 16
    MAX_LENGTH = 256  # 每个包的最大token数
    PACKETS_PER_FLOW = 5  # 每个flow的包数
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    print("\n" + "="*50)
    print("Step 1: Data Preprocessing")
    print("="*50)
    
    preprocessor = USTCTrafficPreprocessor(
        max_length=MAX_LENGTH,
        packet_per_flow=PACKETS_PER_FLOW
    )
    
    # 检查是否存在预处理好的数据
    cache_file = './ustc_et_bert_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            all_samples = pickle.load(f)
    else:
        print("Processing raw pcap files...")
        all_samples = preprocessor.process_dataset(DATA_DIR, CLASS_NAMES)
        with open(cache_file, 'wb') as f:
            pickle.dump(all_samples, f)
        print(f"Saved {len(all_samples)} samples to cache")
    
    print(f"Total samples: {len(all_samples)}")

    if len(all_samples) == 0:
        raise RuntimeError(
            "No samples found. 请检查 DATA_DIR 设置和 pcap 文件是否存在，以及 process_dataset 是否生成了样本。"
        )

    from collections import Counter
    label_counts = Counter([s['label'] for s in all_samples])
    min_count = min(label_counts.values())
    if min_count < 2:
        print("Warning: some classes样本数 < 2，stratify 不能使用，将改为随机拆分。")
        stratify_train = None
    else:
        stratify_train = [s['label'] for s in all_samples]

    # 划分数据集
    train_samples, temp_samples = train_test_split(
        all_samples, test_size=0.2, random_state=42,
        stratify=stratify_train
    )

    label_counts_temp = Counter([s['label'] for s in temp_samples])
    min_count_temp = min(label_counts_temp.values())
    if min_count_temp < 2:
        print("Warning: temp 集合里某些类样本数 < 2，val/test stratify 不能使用，将改为随机拆分。")
        stratify_temp = None
    else:
        stratify_temp = [s['label'] for s in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, random_state=42,
        stratify=stratify_temp
    )
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 创建Dataset和DataLoader
    train_dataset = ET_BERT_Dataset(train_samples, max_length=MAX_LENGTH*PACKETS_PER_FLOW)
    val_dataset = ET_BERT_Dataset(val_samples, max_length=MAX_LENGTH*PACKETS_PER_FLOW)
    test_dataset = ET_BERT_Dataset(test_samples, max_length=MAX_LENGTH*PACKETS_PER_FLOW)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 创建模型
    print("\n" + "="*50)
    print("Step 2: Model Initialization")
    print("="*50)
    
    model = ET_BERT_Classifier(
        num_classes=len(CLASS_NAMES),
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=MAX_LENGTH*PACKETS_PER_FLOW
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 训练
    print("\n" + "="*50)
    print("Step 3: Training")
    print("="*50)
    
    trainer = ET_BERT_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE
    )
    
    history = trainer.train(epochs=EPOCHS)
    
    # 测试
    print("\n" + "="*50)
    print("Step 4: Testing")
    print("="*50)
    
    # 加载最佳模型
    checkpoint = torch.load('./et_bert_checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_labels, test_preds = trainer.evaluate()
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    # 详细报告
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    
    # 保存结果
    results = {
        'history': history,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_labels': test_labels,
        'test_preds': test_preds
    }
    
    with open('./et_bert_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✓ Training completed! Results saved to ./et_bert_results.pkl")


if __name__ == '__main__':
    main()