"""
完整数据预处理脚本 - 适配你的文件结构
支持: CIC-IDS2017 (Parquet/CSV) + USTC-TFC2016 (PCAP/7z)
输出: 4种模型格式 (ResNet, LSTM, CNN+RNN, ET-BERT)
"""

import os
import sys
import pickle
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 可选依赖（处理特定格式）
try:
    from scapy.all import rdpcap, IP, TCP, UDP, Raw
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("警告: 未安装scapy，无法处理PCAP文件")

try:
    import py7zr
    HAS_PY7ZR = True
except ImportError:
    HAS_PY7ZR = False
    print("提示: 安装 py7zr 以处理USTC的.7z文件: pip install py7zr")

# ===================== 路径配置（根据你的结构修改） =====================
BASE_DIR = Path(r"G:\projects\Malicious_Traffic_Detection_System\research")

# CIC-IDS2017 路径（你有Parquet和CSV两种）
CIC_PARQUET_DIR = BASE_DIR / "datasets" / "CIC-IDS2017"  # 包含 *_no-metadata.parquet
CIC_CSV_DIR = BASE_DIR / "datasets" / "CIC-IDS2017" / "MachineLearningCSV"
CIC_PCAP_DIR = BASE_DIR / "datasets" / "CIC-IDS2017" / "pcaps"

# USTC-TFC2016 路径
USTC_DIR = BASE_DIR / "datasets" / "USTC-TFC2016"
USTC_BENIGN = USTC_DIR / "Benign"
USTC_MALWARE = USTC_DIR / "Malware"
USTC_PROCESSED = USTC_DIR / "processed"

# 输出目录
OUTPUT_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型参数
SEQ_LEN = 50        # LSTM/CNN-RNN序列长度
PACKET_BYTES = 256  # 每包字节数（CNN-RNN）
IMG_SIZE = 28       # ResNet图像尺寸（28x28=784字节）


# ===================== 数据结构 =====================
@dataclass
class Packet:
    timestamp: float
    size: int
    payload: bytes
    direction: int


@dataclass
class Flow:
    five_tuple: Tuple
    packets: List[Packet]
    label: str = None


# ===================== 核心处理类 =====================
class DataPreprocessor:
    """统一预处理器"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    # -------------------- CIC-IDS2017 处理 (Parquet优先) --------------------
    def process_cic_from_parquet(self, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从Parquet文件加载（已提取特征，更快）
        适合: ResNet(转图像), LSTM(时序), CNN+RNN(混合)
        """
        print(f"\n[>] 处理 CIC-IDS2017 (Parquet) -> {model_type}")
        
        if not CIC_PARQUET_DIR.exists():
            raise FileNotFoundError(f"找不到目录: {CIC_PARQUET_DIR}")
        
        # 加载所有Parquet文件
        parquet_files = list(CIC_PARQUET_DIR.glob("*-no-metadata.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"在 {CIC_PARQUET_DIR} 中找不到Parquet文件")
        
        print(f"  发现 {len(parquet_files)} 个Parquet文件")
        all_data = []
        
        for pq_file in tqdm(parquet_files, desc="加载Parquet"):
            try:
                df = pd.read_parquet(pq_file)
                # 推断标签从文件名（如 Botnet-Friday -> Botnet）
                label_from_file = pq_file.stem.split('-')[0]  # Benign, Botnet, DDoS等
                if 'Label' not in df.columns:
                    df['Label'] = label_from_file
                all_data.append(df)
            except Exception as e:
                print(f"  错误加载 {pq_file.name}: {e}")
        
        if not all_data:
            raise ValueError("没有成功加载任何Parquet数据")
            
        df = pd.concat(all_data, ignore_index=True)
        print(f"  合并后数据形状: {df.shape}")
        
        # 清洗
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 分离标签
        if 'Label' not in df.columns:
            raise ValueError("数据集中缺少'Label'列")
            
        y = df['Label'].values
        X = df.drop(['Label'], axis=1, errors='ignore')
        
        # 只保留数值列
        X = X.select_dtypes(include=[np.number])
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"  类别: {self.label_encoder.classes_}")
        
        # 根据模型转换格式
        if model_type == 'resnet':
            X_processed = self._to_resnet_format(X.values)
        elif model_type == 'lstm':
            X_processed = self._to_lstm_format(X.values)
        elif model_type == 'cnn_rnn':
            X_processed = self._to_cnn_rnn_format(X.values)
        elif model_type == 'etbert':
            X_processed = self._to_etbert_format(X.values)
        else:
            raise ValueError(f"不支持的模型: {model_type}")
            
        return X_processed, y_encoded
    
    def process_cic_from_csv(self, csv_path: Path, model_type: str):
        """从CSV加载（备用方案）"""
        print(f"[>] 从CSV加载: {csv_path.name}")
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        y = df['Label'].values
        X = df.drop(['Label'], axis=1).select_dtypes(include=[np.number])
        y_encoded = self.label_encoder.fit_transform(y)
        
        if model_type == 'resnet':
            X_processed = self._to_resnet_format(X.values)
        elif model_type == 'lstm':
            X_processed = self._to_lstm_format(X.values)
        elif model_type == 'cnn_rnn':
            X_processed = self._to_cnn_rnn_format(X.values)
        elif model_type == 'etbert':
            X_processed = self._to_etbert_format(X.values)
            
        return X_processed, y_encoded
    
    # -------------------- USTC-TFC2016 处理 (PCAP + 7z) --------------------
    def process_ustc(self, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理USTC数据集（解压7z + 解析PCAP）
        20类分类任务（10正常 + 10恶意）
        """
        print(f"\n[>] 处理 USTC-TFC2016 -> {model_type}")
        
        # 检查并解压7z文件
        self._extract_7z_files()
        
        # 收集所有PCAP
        pcap_files = []
        labels = []
        
        # Benign类别（10类）
        if USTC_BENIGN.exists():
            for pcap in USTC_BENIGN.glob("*.pcap"):
                pcap_files.append(pcap)
                labels.append(f"Benign-{pcap.stem}")  # Benign-BitTorrent等
        
        # Malware类别（10类）
        if USTC_MALWARE.exists():
            for pcap in USTC_MALWARE.glob("*.pcap"):
                pcap_files.append(pcap)
                labels.append(f"Malware-{pcap.stem}")  # Malware-Cridex等
        
        if not pcap_files:
            raise FileNotFoundError(f"在 {USTC_DIR} 中找不到PCAP文件（请先解压.7z）")
        
        print(f"  发现 {len(pcap_files)} 个PCAP文件")
        
        # 解析PCAP为Flow
        all_features = []
        all_labels = []
        
        for pcap_file, label in tqdm(zip(pcap_files, labels), total=len(pcap_files), desc="解析PCAP"):
            try:
                if model_type == 'resnet':
                    feat = self._pcap_to_resnet_image(pcap_file)
                elif model_type == 'lstm':
                    feat = self._pcap_to_lstm_sequence(pcap_file)
                elif model_type == 'cnn_rnn':
                    feat = self._pcap_to_cnn_rnn_matrix(pcap_file)
                elif model_type == 'etbert':
                    feat = self._pcap_to_etbert_tokens(pcap_file)
                else:
                    continue
                
                if feat is not None:
                    all_features.append(feat)
                    all_labels.append(label)
            except Exception as e:
                print(f"  处理 {pcap_file.name} 失败: {e}")
                continue
        
        X = np.array(all_features)
        y = self.label_encoder.fit_transform(all_labels)
        
        print(f"  成功处理: {len(X)} 个样本")
        print(f"  类别数: {len(self.label_encoder.classes_)}")
        
        return X, y
    
    def _extract_7z_files(self):
        """自动解压USTC中的7z文件"""
        if not HAS_PY7ZR:
            return
            
        seven_z_files = list(USTC_DIR.rglob("*.7z"))
        if not seven_z_files:
            return
            
        print(f"  发现 {len(seven_z_files)} 个.7z文件，正在解压...")
        for zfile in tqdm(seven_z_files, desc="解压"):
            try:
                with py7zr.SevenZipFile(zfile, mode='r') as z:
                    z.extractall(path=zfile.parent)
                print(f"    已解压: {zfile.name}")
            except Exception as e:
                print(f"    解压失败 {zfile.name}: {e}")
    
    # -------------------- 格式转换方法 --------------------
    def _to_resnet_format(self, X: np.ndarray) -> np.ndarray:
        """表格数据 -> 28x28灰度图像"""
        n_samples = X.shape[0]
        target_size = IMG_SIZE * IMG_SIZE
        
        # 归一化到[0,1]
        X_norm = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8)
        
        # 填充或截断到784特征
        if X_norm.shape[1] < target_size:
            padding = np.zeros((n_samples, target_size - X_norm.shape[1]))
            X_padded = np.hstack([X_norm, padding])
        else:
            X_padded = X_norm[:, :target_size]
        
        # 重塑为图像 (N, 1, 28, 28)
        X_img = X_padded.reshape(n_samples, 1, IMG_SIZE, IMG_SIZE)
        return X_img.astype(np.float32)
    
    def _to_lstm_format(self, X: np.ndarray) -> np.ndarray:
        """表格数据 -> 时序序列 (N, 50, 2)"""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # 归一化
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # 分成50个时间步，每步2个特征（简化处理：前100维分为50x2）
        seq_len = SEQ_LEN
        n_per_step = min(2, n_features // seq_len)
        
        if n_features >= seq_len * 2:
            X_seq = X_norm[:, :seq_len*2].reshape(n_samples, seq_len, 2)
        else:
            # 填充
            padded = np.zeros((n_samples, seq_len * 2))
            padded[:, :n_features] = X_norm
            X_seq = padded.reshape(n_samples, seq_len, 2)
            
        return X_seq.astype(np.float32)
    
    def _to_cnn_rnn_format(self, X: np.ndarray) -> np.ndarray:
        """表格数据 -> 字节矩阵 (N, 50, 256)"""
        n_samples = X.shape[0]
        
        # 归一化到字节范围[0,255]
        X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)
        X_bytes = (X_norm * 255).astype(np.uint8)
        
        # 重塑为 (N, 50, 256) - 取前12800个特征
        target_flat = SEQ_LEN * PACKET_BYTES  # 50*256=12800
        if X_bytes.shape[1] < target_flat:
            padded = np.zeros((n_samples, target_flat), dtype=np.uint8)
            padded[:, :X_bytes.shape[1]] = X_bytes
            X_bytes = padded
        else:
            X_bytes = X_bytes[:, :target_flat]
            
        X_matrix = X_bytes.reshape(n_samples, SEQ_LEN, PACKET_BYTES)
        return X_matrix.astype(np.float32) / 255.0  # 归一化到[0,1]
    
    def _to_etbert_format(self, X: np.ndarray) -> np.ndarray:
        """表格数据 -> 类字节序列 (N, 512)"""
        # 模拟字节序列：归一化到0-255
        X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)
        X_bytes = (X_norm * 255).astype(np.uint8)
        
        # 截断或填充到512
        if X_bytes.shape[1] > 512:
            return X_bytes[:, :512]
        else:
            padding = np.zeros((X_bytes.shape[0], 512 - X_bytes.shape[1]), dtype=np.uint8)
            return np.hstack([X_bytes, padding])
    
    # -------------------- PCAP直接解析（USTC用） --------------------
    def _pcap_to_resnet_image(self, pcap_path: Path) -> np.ndarray:
        """PCAP -> 28x28图像（取前784字节Payload）"""
        if not HAS_SCAPY:
            raise ImportError("需要scapy处理PCAP")
            
        packets = rdpcap(str(pcap_path))
        all_bytes = b''
        
        for pkt in packets[:20]:  # 取前20个包
            if Raw in pkt:
                all_bytes += bytes(pkt[Raw])
            if len(all_bytes) >= 784:
                break
        
        # 填充到784字节
        if len(all_bytes) < 784:
            all_bytes += b'\x00' * (784 - len(all_bytes))
        all_bytes = all_bytes[:784]
        
        img = np.frombuffer(all_bytes, dtype=np.uint8).reshape(28, 28)
        img = (img / 127.5) - 1.0  # 归一化到[-1,1]
        return img[np.newaxis, :, :]  # (1, 28, 28)
    
    def _pcap_to_lstm_sequence(self, pcap_path: Path) -> np.ndarray:
        """PCAP -> 时序特征 (50, 2) [size, iat]"""
        if not HAS_SCAPY:
            raise ImportError("需要scapy处理PCAP")
            
        packets = rdpcap(str(pcap_path))
        
        sizes = []
        iats = []
        prev_ts = None
        
        for pkt in packets[:SEQ_LEN]:
            if IP in pkt:
                size = len(pkt)
                ts = float(pkt.time)
                iat = 0 if prev_ts is None else ts - prev_ts
                prev_ts = ts
                
                sizes.append(size)
                iats.append(iat)
        
        # 填充
        while len(sizes) < SEQ_LEN:
            sizes.append(0)
            iats.append(0)
        
        sizes = np.array(sizes, dtype=np.float32)
        iats = np.array(iats, dtype=np.float32)
        
        # 标准化
        sizes = (sizes - sizes.mean()) / (sizes.std() + 1e-8)
        iats = np.clip(iats, 0, 10) / 10.0
        
        return np.column_stack([sizes, iats])  # (50, 2)
    
    def _pcap_to_cnn_rnn_matrix(self, pcap_path: Path) -> np.ndarray:
        """PCAP -> 字节矩阵 (50, 256)"""
        if not HAS_SCAPY:
            raise ImportError("需要scapy处理PCAP")
            
        packets = rdpcap(str(pcap_path))
        matrix = np.zeros((SEQ_LEN, PACKET_BYTES), dtype=np.float32)
        
        for i, pkt in enumerate(packets[:SEQ_LEN]):
            if Raw in pkt:
                payload = bytes(pkt[Raw])[:PACKET_BYTES]
                if len(payload) < PACKET_BYTES:
                    payload += b'\x00' * (PACKET_BYTES - len(payload))
                matrix[i] = np.array([b/255.0 for b in payload])
        
        return matrix  # (50, 256)
    
    def _pcap_to_etbert_tokens(self, pcap_path: Path) -> str:
        """PCAP -> Hex字符串"""
        if not HAS_SCAPY:
            raise ImportError("需要scapy处理PCAP")
            
        packets = rdpcap(str(pcap_path))
        all_bytes = b''.join([bytes(pkt[Raw]) for pkt in packets[:10] if Raw in pkt])
        
        # 转为Hex字符串，截断到512字节
        all_bytes = all_bytes[:512]
        return all_bytes.hex()
    
    # -------------------- 保存与加载 --------------------
    def save_processed(self, X: np.ndarray, y: np.ndarray, model_type: str, dataset_name: str):
        """保存处理好的数据"""
        output_file = OUTPUT_DIR / f"{dataset_name}_{model_type}_data.npz"
        np.savez(output_file, X=X, y=y, classes=self.label_encoder.classes_)
        print(f"[✓] 已保存: {output_file}")
        print(f"    数据形状: {X.shape}, 标签形状: {y.shape}")
        return output_file


# ===================== 主函数 =====================
def main():
    """主入口"""
    print("="*60)
    print("恶意流量检测 - 数据预处理")
    print("支持: CIC-IDS2017 (Parquet/CSV) | USTC-TFC2016 (PCAP/7z)")
    print("输出: ResNet | LSTM | CNN+RNN | ET-BERT 格式")
    print("="*60)
    
    processor = DataPreprocessor()
    
    # 处理CIC-IDS2017 (优先使用Parquet，快且已清洗)
    try:
        for model in ['resnet', 'lstm', 'cnn_rnn', 'etbert']:
            try:
                X, y = processor.process_cic_from_parquet(model)
                processor.save_processed(X, y, model, "CIC")
            except Exception as e:
                print(f"[✗] CIC {model} 处理失败: {e}")
                # 尝试CSV备用
                if (CIC_CSV_DIR / "Wednesday-workingHours.pcap_ISCX.csv").exists():
                    print("  尝试从CSV加载...")
                    X, y = processor.process_cic_from_csv(
                        CIC_CSV_DIR / "Wednesday-workingHours.pcap_ISCX.csv", 
                        model
                    )
                    processor.save_processed(X, y, model, "CIC")
    except Exception as e:
        print(f"[✗] CIC数据集处理失败: {e}")
    
    # 处理USTC-TFC2016 (需要scapy和py7zr)
    try:
        if HAS_SCAPY:
            for model in ['resnet', 'lstm', 'cnn_rnn', 'etbert']:
                try:
                    X, y = processor.process_ustc(model)
                    processor.save_processed(X, y, model, "USTC")
                except Exception as e:
                    print(f"[✗] USTC {model} 处理失败: {e}")
        else:
            print("[!] 跳过USTC处理（未安装scapy）")
    except Exception as e:
        print(f"[✗] USTC数据集处理失败: {e}")
    
    print("\n" + "="*60)
    print("预处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("文件列表:")
    for f in OUTPUT_DIR.glob("*.npz"):
        print(f"  - {f.name}")
    print("="*60)


if __name__ == "__main__":
    main()