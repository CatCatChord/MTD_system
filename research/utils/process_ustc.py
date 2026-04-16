#!/usr/bin/env python3
"""
USTC-TFC2016 数据预处理（修复版）
关键修改：
1. LSTM: 3维->15维（添加统计特征）
2. ResNet: 39x39->64x64（3通道，适配ImageNet预训练）
3. CNN-RNN: float32->int64（使用nn.Embedding替代直接归一化）
4. 标签: 20分类->2分类（正常vs恶意）
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from scapy.all import rdpcap, IP, TCP, UDP, Raw

# ===================== 配置 =====================
USTC_DIR = Path("/mnt/data/ckb_datasets/USTC-TFC2016")
BENIGN_DIR = USTC_DIR / "Benign"
MALWARE_DIR = USTC_DIR / "Malware"
OUTPUT_DIR = Path("/mnt/data/processed_data_fixed")  # 新目录，避免覆盖旧数据
OUTPUT_DIR.mkdir(exist_ok=True)

# FIX 1: 增大序列长度和图像尺寸
SEQ_LEN = 50              
PACKET_BYTES = 256        
IMG_SIZE = 64              # FIX: 从39改为64（适配ResNet预训练）
MAX_FLOW_PER_PCAP = 500   
MIN_PACKETS = 5           

# 类别映射（20类，但输出改为2分类）
BENIGN_CLASSES = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 
                  'Outlook', 'Skype', 'SMB', 'Weibo', 'WorldOfWarcraft']
MALWARE_CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 
                   'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']


class USTCPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.flow_stats = defaultdict(int)
        
    def process(self, model_type: str):
        """主处理"""
        print(f"\n{'='*60}")
        print(f"处理 USTC-TFC2016 -> {model_type.upper()} [修复版]")
        print(f"数据路径: {USTC_DIR}")
        print(f"输出路径: {OUTPUT_DIR}")
        print(f"{'='*60}")
        
        pcap_files = self._collect_all_pcaps()
        if not pcap_files:
            raise FileNotFoundError(f"在 {USTC_DIR} 中没有找到PCAP文件！")
        
        print(f"[1/4] 发现 {len(pcap_files)} 个PCAP文件")
        
        all_features = []
        all_labels = []
        
        for pcap_path in tqdm(pcap_files, desc="处理PCAP"):
            try:
                # FIX 2: 改为二分类标签（0=正常，1=恶意）
                label = self._get_binary_label(pcap_path)
                if label is None:
                    continue
                
                flows = self._extract_flows(pcap_path, max_flows=MAX_FLOW_PER_PCAP)
                
                for flow in flows:
                    feat = self._flow_to_feature(flow, model_type)
                    if feat is not None:
                        all_features.append(feat)
                        all_labels.append(label)
                        self.flow_stats[label] += 1
                        
            except Exception as e:
                print(f"\n  跳过 {pcap_path.name}: {e}")
                continue
        
        if len(all_features) == 0:
            raise ValueError("没有提取到任何有效特征！")
        
        print(f"\n[2/4] 总共提取 {len(all_features)} 个流样本")
        print("类别分布：")
        for cls, cnt in sorted(self.flow_stats.items()):
            print(f"    {'Benign' if cls==0 else 'Malware'}: {cnt}")
        
        X = np.array(all_features)
        y = np.array(all_labels)  # FIX: 直接使用0/1标签，无需encoder
        
        print(f"\n[3/4] 数据形状: {X.shape}, 正样本比例: {np.mean(y):.2%}")
        
        # FIX 3: 使用分层抽样确保训练/测试集类别比例一致
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self._save(X_train, X_test, y_train, y_test, model_type)
        return X_train, X_test, y_train, y_test
    
    def _collect_all_pcaps(self) -> List[Path]:
        """收集PCAP（保持原逻辑）"""
        pcaps = []
        
        if BENIGN_DIR.exists():
            for cls in BENIGN_CLASSES:
                matches = list(BENIGN_DIR.glob(f"{cls}*.pcap"))
                pcaps.extend(matches)
                if matches:
                    print(f"  [良性] {cls}: {len(matches)} 个文件")
        
        if MALWARE_DIR.exists():
            for cls in MALWARE_CLASSES:
                matches = list(MALWARE_DIR.glob(f"{cls}*.pcap"))
                pcaps.extend(matches)
                if matches:
                    print(f"  [恶意] {cls}: {len(matches)} 个文件")
        
        return sorted(set(pcaps))
    
    def _get_binary_label(self, pcap_path: Path) -> int:
        """
        FIX 4: 二分类标签
        0 = Benign (正常流量)
        1 = Malware (恶意流量)
        """
        name = pcap_path.stem.lower()
        
        for cls in BENIGN_CLASSES:
            if cls.lower() in name:
                return 0  # 正常
        
        for cls in MALWARE_CLASSES:
            if cls.lower() in name:
                return 1  # 恶意
        
        return None
    
    def _extract_flows(self, pcap_path: Path, max_flows: int = 500) -> List[Dict]:
        """提取流（保持原逻辑，但增加时间戳处理）"""
        packets = rdpcap(str(pcap_path))
        flows = defaultdict(list)
        
        for pkt in packets:
            if IP not in pkt:
                continue
            
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                proto = 'TCP'
                payload = bytes(pkt[TCP].payload) if Raw in pkt else b''
                # FIX: 尝试提取TCP标志位（如果有）
                flags = pkt[TCP].flags if hasattr(pkt[TCP], 'flags') else 0
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                proto = 'UDP'
                payload = bytes(pkt[UDP].payload) if Raw in pkt else b''
                flags = 0
            else:
                continue
            
            # 五元组作为流键
            if (src_ip, src_port) < (dst_ip, dst_port):
                key = (src_ip, dst_ip, src_port, dst_port, proto)
                direction = 0  # 上行
            else:
                key = (dst_ip, src_ip, dst_port, src_port, proto)
                direction = 1  # 下行
            
            flows[key].append({
                'timestamp': float(pkt.time),
                'size': len(pkt),
                'payload': payload,
                'direction': direction,
                'flags': int(flags)
            })
        
        # 构建有效流
        valid_flows = []
        for key, pkts in flows.items():
            if len(pkts) < MIN_PACKETS:
                continue
            
            # 按时间排序
            pkts.sort(key=lambda x: x['timestamp'])
            pkts = pkts[:SEQ_LEN]  # 截断到最大长度
            
            # 计算IAT（包间到达时间）
            for i in range(1, len(pkts)):
                pkts[i]['iat'] = pkts[i]['timestamp'] - pkts[i-1]['timestamp']
            pkts[0]['iat'] = 0.0
            
            valid_flows.append({'packets': pkts, 'key': key})
        
        # 随机采样限制数量
        if len(valid_flows) > max_flows:
            random.seed(42)
            valid_flows = random.sample(valid_flows, max_flows)
        
        return valid_flows
    
    def _flow_to_feature(self, flow: Dict, model_type: str):
        """根据模型类型提取特征"""
        try:
            pkts = flow['packets']
            if model_type == 'resnet':
                return self._to_resnet(pkts)
            elif model_type == 'lstm':
                return self._to_lstm(pkts)
            elif model_type == 'cnn_rnn':
                return self._to_cnn_rnn(pkts)
        except Exception as e:
            return None
    
    def _to_resnet(self, pkts: List[Dict]):
        """
        FIX 5: ResNet输入改为64x64，3通道，ImageNet标准化
        输出: [3, 64, 64] (C, H, W)
        """
        # 拼接所有包的payload（最多取前4096字节 = 64*64）
        all_bytes = b''.join([p['payload'] for p in pkts])
        target_len = IMG_SIZE * IMG_SIZE  # 4096字节
        
        if len(all_bytes) < target_len:
            all_bytes += b'\x00' * (target_len - len(all_bytes))
        else:
            all_bytes = all_bytes[:target_len]
        
        # 转为图像 [64, 64]
        img = np.frombuffer(all_bytes, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        img = img.astype(np.float32)
        
        # ImageNet风格标准化（适配预训练ResNet）
        # 注意：如果不用预训练权重，可以只用 (img - 128) / 128
        img = (img - 128.0) / 128.0  # 归一化到[-1, 1]
        
        # 复制为3通道 [3, 64, 64]
        img = np.stack([img, img, img], axis=0)
        return img
    
    def _to_lstm(self, pkts: List[Dict]):
        """
        FIX 6: LSTM输入从3维扩展到15维（时序+统计特征）
        输出: [SEQ_LEN, 15]
        """
        # 基础时序特征（3维）
        sizes = np.array([p['size'] for p in pkts], dtype=np.float32)
        iats = np.array([p['iat'] for p in pkts], dtype=np.float32)
        dirs = np.array([p['direction'] for p in pkts], dtype=np.float32)
        
        # 填充到SEQ_LEN
        if len(sizes) < SEQ_LEN:
            sizes = np.pad(sizes, (0, SEQ_LEN - len(sizes)), mode='constant')
            iats = np.pad(iats, (0, SEQ_LEN - len(iats)), mode='constant', constant_values=0)
            dirs = np.pad(dirs, (0, SEQ_LEN - len(dirs)), mode='constant', constant_values=-1)
        
        # FIX 7: 添加全局统计特征（12维），在时序维度上广播
        if len(pkts) > 0:
            # 包长统计
            size_mean = np.mean(sizes[:len(pkts)])
            size_std = np.std(sizes[:len(pkts)]) if len(pkts) > 1 else 0
            size_min = np.min(sizes[:len(pkts)])
            size_max = np.max(sizes[:len(pkts)])
            
            # IAT统计（排除第一个0）
            valid_iats = iats[1:len(pkts)]
            iat_mean = np.mean(valid_iats) if len(valid_iats) > 0 else 0
            iat_std = np.std(valid_iats) if len(valid_iats) > 1 else 0
            iat_max = np.max(valid_iats) if len(valid_iats) > 0 else 0
            
            # 方向统计
            dir_0_ratio = np.sum(dirs[:len(pkts)] == 0) / len(pkts)  # 上行比例
            dir_1_ratio = 1 - dir_0_ratio
            
            # 字节量统计
            total_bytes = np.sum(sizes[:len(pkts)])
            bytes_up = np.sum(sizes[:len(pkts)] * (dirs[:len(pkts)] == 0))
            bytes_down = total_bytes - bytes_up
            
            # 流持续时间
            duration = np.sum(iats[:len(pkts)])
            
            # 包数量（归一化）
            pkt_count = len(pkts) / SEQ_LEN
            
            stat_features = np.array([
                size_mean, size_std, size_min, size_max,
                iat_mean, iat_std, iat_max,
                dir_0_ratio, dir_1_ratio,
                bytes_up / 1000.0, bytes_down / 1000.0,  # KB为单位
                duration, pkt_count
            ])
        else:
            stat_features = np.zeros(12)
        
        # 拼接: [SEQ_LEN, 3] + [SEQ_LEN, 12] = [SEQ_LEN, 15]
        seq_features = np.column_stack([sizes, iats, dirs])
        stat_repeated = np.tile(stat_features, (SEQ_LEN, 1))
        
        return np.concatenate([seq_features, stat_repeated], axis=1)
    
    def _to_cnn_rnn(self, pkts: List[Dict]):
        """
        FIX 8: CNN-RNN输入改为int64（用于nn.Embedding）
        输出: [SEQ_LEN, PACKET_BYTES]，dtype=int64
        """
        # 构建矩阵 [SEQ_LEN, PACKET_BYTES]
        matrix = np.zeros((SEQ_LEN, PACKET_BYTES), dtype=np.int64)  # 关键：int64而非float32
        
        for i, p in enumerate(pkts[:SEQ_LEN]):
            payload = p['payload'][:PACKET_BYTES]
            if len(payload) < PACKET_BYTES:
                payload += b'\x00' * (PACKET_BYTES - len(payload))
            
            # 保持0-255的整数值，不除255！
            matrix[i] = np.array([b for b in payload], dtype=np.int64)
        
        return matrix
    
    def _save(self, X_train, X_test, y_train, y_test, model_type):
        """保存数据"""
        output_file = OUTPUT_DIR / f"USTC_{model_type}_fixed.npz"
        np.savez_compressed(
            output_file,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            # FIX 9: 保存类别信息（二分类）
            classes=np.array(['Benign', 'Malware'])
        )
        print(f"\n[✓] 已保存: {output_file}")
        print(f"    训练集: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
        print(f"    测试集: {X_test.shape}, 标签分布: {np.bincount(y_test)}")


def verify_data():
    """验证修复后的数据"""
    print("\n" + "="*60)
    print("验证修复后的数据")
    print("="*60)
    
    for model in ['resnet', 'lstm', 'cnn_rnn']:
        file = OUTPUT_DIR / f"USTC_{model}_fixed.npz"
        if not file.exists():
            print(f"[✗] {model}: 不存在")
            continue
        
        data = np.load(file)
        X_train = data['X_train']
        y_train = data['y_train']
        
        print(f"\n{model.upper()}:")
        print(f"  形状: {X_train.shape}, 类型: {X_train.dtype}")
        print(f"  类别: {data['classes']}")
        print(f"  正样本比例: {np.mean(y_train):.2%}")
        
        # 显示一个样本的统计
        if model == 'lstm':
            print(f"  样本均值: {np.mean(X_train[0]):.4f}, 标准差: {np.std(X_train[0]):.4f}")
        elif model == 'resnet':
            print(f"  值范围: [{np.min(X_train):.2f}, {np.max(X_train):.2f}]")
        elif model == 'cnn_rnn':
            print(f"  唯一值数量: {len(np.unique(X_train))} (应为256表示0-255字节)")


def main():
    # 清理旧文件
    old_files = list(OUTPUT_DIR.glob("USTC_*_fixed.npz"))
    if old_files:
        print(f"发现 {len(old_files)} 个旧文件")
        if input("删除旧文件？[y/N]: ").lower() == 'y':
            for f in old_files:
                f.unlink()
    
    processor = USTCPreprocessor()
    for model in ['resnet', 'lstm', 'cnn_rnn']:
        try:
            processor.process(model)
            print()
        except Exception as e:
            print(f"\n[✗] 失败: {e}")
            import traceback
            traceback.print_exc()
    
    verify_data()


if __name__ == "__main__":
    main()