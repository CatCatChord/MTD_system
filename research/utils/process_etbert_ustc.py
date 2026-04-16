#!/usr/bin/env python3
"""
USTC-TFC2016 包级预处理（修正版）- 适配真实结构
"""

import os
import numpy as np
from scapy.all import *
import binascii
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import csv
from tqdm import tqdm
from collections import defaultdict
import subprocess
import tempfile
import shutil

PCAP_ROOT = Path("/mnt/data/ckb_datasets/USTC-TFC2016")
OUTPUT_DIR = Path("/mnt/data/ckb_datasets/processed_data_etbert_packet_complete")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 20类配置（与之前相同）
CATEGORY_CONFIG = [
    ("BitTorrent", "Benign", False), ("Facetime", "Benign", False),
    ("FTP", "Benign", False), ("Gmail", "Benign", False),
    ("MySQL", "Benign", False), ("Outlook", "Benign", False),
    ("Skype", "Benign", False), ("SMB", "Benign", True),
    ("Weibo", "Benign", True), ("WorldOfWarcraft", "Benign", False),
    ("Cridex", "Malware", True), ("Geodo", "Malware", True),
    ("Htbot", "Malware", True), ("Miuref", "Malware", False),
    ("Neris", "Malware", True), ("Nsis-ay", "Malware", True),
    ("Shifu", "Malware", True), ("Tinba", "Malware", False),
    ("Virut", "Malware", True), ("Zeus", "Malware", False),
]

SAMPLES_PER_CLASS = 5000
MAX_BYTES = 256  # 包级用256字节

def extract_payload(pkt, max_len=MAX_BYTES):
    try:
        if IP in pkt:
            payload = bytes(pkt[TCP].payload) if TCP in pkt else \
                     bytes(pkt[UDP].payload) if UDP in pkt else bytes(pkt[IP].payload)
            if len(payload) < 40:
                return None
            if len(payload) > max_len:
                payload = payload[:max_len]
            else:
                payload = payload + b'\x00' * (max_len - len(payload))
            return payload
    except:
        return None

def bytes_to_hex_tokens(payload):
    hex_str = binascii.hexlify(payload).decode()
    return ' '.join([hex_str[i:i+4] for i in range(0, len(hex_str), 4)])

def extract_7z(archive_path, extract_to):
    try:
        result = subprocess.run(['7z', 'x', str(archive_path), f'-o{extract_to}', '-y'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def process_category(label_id, class_name, parent, has_7z):
    """处理单个类别，返回样本列表"""
    sources = []
    base_path = PCAP_ROOT / parent / class_name
    
    # 收集所有PCAP来源
    if base_path.with_suffix('.pcap').exists():
        sources.append(base_path.with_suffix('.pcap'))
    
    if has_7z:
        archive = base_path.with_suffix('.7z')
        if archive.exists():
            temp_dir = tempfile.mkdtemp()
            if extract_7z(archive, temp_dir):
                sources.extend(list(Path(temp_dir).rglob("*.pcap")))
    
    if base_path.is_dir():
        sources.extend(list(base_path.rglob("*.pcap")))
    
    if not sources:
        return []
    
    # 提取包（单包，不是流）
    samples = []
    for pcap_path in sources:
        if len(samples) >= SAMPLES_PER_CLASS:
            break
        try:
            for pkt in rdpcap(str(pcap_path)):
                if len(samples) >= SAMPLES_PER_CLASS:
                    break
                payload = extract_payload(pkt, MAX_BYTES)
                if payload:
                    hex_tokens = bytes_to_hex_tokens(payload)
                    samples.append((label_id, hex_tokens))
        except Exception as e:
            continue
    
    return samples[:SAMPLES_PER_CLASS]

def main():
    print("USTC-TFC2016 包级预处理（真实结构适配版）")
    print(f"输出: {OUTPUT_DIR}")
    
    all_data = []
    class_counts = [0] * 20
    temp_dirs = []
    
    for label_id, (class_name, parent, has_7z) in enumerate(CATEGORY_CONFIG):
        print(f"\n处理类别 {label_id} ({class_name})...")
        samples = process_category(label_id, class_name, parent, has_7z)
        all_data.extend(samples)
        class_counts[label_id] = len(samples)
        print(f"  ✅ 提取 {len(samples)}/{SAMPLES_PER_CLASS} 个包")
    
    # 清理临时目录
    for f in Path(tempfile.gettempdir()).glob("*.pcap"):
        try:
            f.unlink()
        except:
            pass
    
    # 划分数据集
    print(f"\n总计: {len(all_data)} 个包样本")
    print(f"分布: {class_counts}")
    
    if len(all_data) == 0:
        print("❌ 无数据")
        return
    
    X = [x[1] for x in all_data]
    y = [x[0] for x in all_data]
    
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, tmp = next(split1.split(X, y))
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    va, te = next(split2.split([X[i] for i in tmp], [y[i] for i in tmp]))
    
    # 保存
    def save(X_data, y_data, fname):
        with open(OUTPUT_DIR / fname, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["label", "text_a"])
            for label, text in zip(y_data, X_data):
                writer.writerow([label, text])
    
    save([X[i] for i in tr], [y[i] for i in tr], "train_dataset.tsv")
    save([X[i] for i in [tmp[i] for i in va]], [y[i] for i in [tmp[i] for i in va]], "valid_dataset.tsv")
    save([X[i] for i in [tmp[i] for i in te]], [y[i] for i in [tmp[i] for i in te]], "test_dataset.tsv")
    
    print(f"\n✅ 完成!")
    print(f"使用 --max_seq_length 128 训练")
    print(f"命令: python fine-tuning/run_classifier.py --train_path {OUTPUT_DIR}/train_dataset.tsv --max_seq_length 128 --epochs_num 20 ...")

if __name__ == "__main__":
    main()