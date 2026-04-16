#!/usr/bin/env python3
"""
CIC-IDS2017 纯CSV预处理（分层采样修复版）
- 修复Label列检测
- 支持周一到周五所有CSV自动检测
- CNN+RNN使用StratifiedShuffleSplit分层采样（保持类别比例）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ===================== 配置路径 =====================
BASE_DIR = Path(r"G:\projects\Malicious_Traffic_Detection_System\research")
CIC_CSV_DIR = BASE_DIR / "datasets" / "CIC-IDS2017" / "MachineLearningCSV"
OUTPUT_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
SEQ_LEN = 50
PACKET_BYTES = 256


class CICPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.supported_models = ['resnet', 'lstm', 'cnn_rnn']
        # 新增：存储全局统计量
        self.feature_means = None
        self.feature_stds = None
        self.feature_indices = None  # 选择的特征索引
        
    def process(self, model_type: str):
        """主处理入口"""
        if model_type not in self.supported_models:
            raise ValueError(f"不支持模型 {model_type}")
        
        print(f"\n{'='*60}")
        print(f"处理 CIC-IDS2017 -> {model_type.upper()}")
        print(f"{'='*60}")
        
        df = self._load_from_csv()
        df = self._clean_data(df)
        X, y = self._extract_features_labels(df)
        
        # 智能采样：确保每类最小样本数 + 总上限
        if model_type == 'cnn_rnn':
            MAX_TOTAL = 100000
            MIN_PER_CLASS = 50  # 每类至少50个，确保能学习
        else:  # resnet/lstm
            MAX_TOTAL = 500000
            MIN_PER_CLASS = 100  # 每类至少100个
        
        if len(X) > MAX_TOTAL:
            print(f"\n  [智能分层采样] 原数据{len(X)}行，确保每类≥{MIN_PER_CLASS}个样本")
            
            # 步骤1: 先确保每类至少保留MIN_PER_CLASS个（优先保障罕见类）
            unique_classes = np.unique(y)
            selected_indices = []
            
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                
                if len(cls_indices) >= MIN_PER_CLASS:
                    # 该类样本充足，随机采MIN_PER_CLASS个
                    selected = np.random.choice(cls_indices, MIN_PER_CLASS, replace=False)
                else:
                    # 该类样本不足，全保留（并警告）
                    selected = cls_indices
                    print(f"    ⚠️ 类别{cls}只有{len(cls_indices)}个样本（低于{MIN_PER_CLASS}）")
                
                selected_indices.extend(selected)
            
            # 步骤2: 剩余配额按原比例分配给大类（提升总体性能）
            current_count = len(selected_indices)
            if current_count < MAX_TOTAL:
                remaining = MAX_TOTAL - current_count
                # 从剩余样本中随机采remaining个
                mask = np.ones(len(X), dtype=bool)
                mask[selected_indices] = False
                other_indices = np.where(mask)[0]
                
                if len(other_indices) > 0:
                    additional = np.random.choice(other_indices, min(remaining, len(other_indices)), replace=False)
                    selected_indices.extend(additional)
            
            selected_indices = np.array(selected_indices)
            X = X[selected_indices]
            y = y[selected_indices]
            
            # 验证结果
            unique, counts = np.unique(y, return_counts=True)
            print(f"  采样完成: 共{len(X)}个样本")
            print(f"  类别分布: 最少{counts.min()}个, 最多{counts.max()}个")
            if counts.min() < MIN_PER_CLASS:
                print(f"  ⚠️ 警告: {len(counts[counts < MIN_PER_CLASS])}个类别样本不足{MIN_PER_CLASS}")
        
        X_processed = self._convert_format(X, model_type)
        
        # 划分数据集（检查stratify）
        unique, counts = np.unique(y, return_counts=True)
        if np.min(counts) < 2:
            print(f"  [警告] 某类别只有{np.min(counts)}个样本，禁用分层抽样")
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
        
        self._save(X_train, X_test, y_train, y_test, model_type)
        return X_train, X_test, y_train, y_test
    
    def _load_from_csv(self) -> pd.DataFrame:
        """加载周一到周五所有存在的CSV文件"""
        print("[1/4] 从CSV加载数据...")
        
        if not CIC_CSV_DIR.exists():
            raise FileNotFoundError(f"CSV目录不存在: {CIC_CSV_DIR}")
        
        csv_candidates = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv", 
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
        
        dfs = []
        found_files = []
        
        for csv_name in csv_candidates:
            csv_path = CIC_CSV_DIR / csv_name
            if csv_path.exists():
                try:
                    print(f"  正在加载: {csv_name} ...")
                    df = pd.read_csv(csv_path, low_memory=False)
                    dfs.append(df)
                    found_files.append(csv_name)
                    print(f"    ✓ 成功: {len(df)} 行, {len(df.columns)} 列")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")
        
        if not dfs:
            raise ValueError(f"在 {CIC_CSV_DIR} 中没有找到任何CSV文件！")
        
        print(f"\n  共加载 {len(found_files)} 个文件:")
        for f in found_files:
            print(f"    - {f}")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"\n  合并完成: {df.shape[0]} 行, {df.shape[1]} 列")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        print("[2/4] 数据清洗...")
        initial = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        cols_to_drop = ['Source IP', 'Destination IP', 'Source Port', 
                       'Destination Port', 'Protocol', 'Timestamp', 'Flow ID',
                       'Source IP', 'Destination IP']
        existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(existing_drop_cols, axis=1, errors='ignore')
        
        print(f"  清洗: {initial} -> {len(df)} 行")
        return df
    
    def _extract_features_labels(self, df: pd.DataFrame):
        """提取X, y"""
        print("[3/4] 提取特征...")
        
        label_col = None
        possible_names = ['Label', ' label', 'Label ', 'label', 'LABEL']
        
        for col in df.columns:
            if col.strip() in possible_names or col.lower() == 'label':
                label_col = col
                break
        
        if label_col is None:
            print(f"\n  [错误] 可用列名: {list(df.columns)}")
            raise ValueError(f"找不到Label列！尝试了: {possible_names}")
        
        print(f"  使用标签列: '{label_col}'")
        
        y = df[label_col].values
        X = df.drop([label_col], axis=1)
        X = X.select_dtypes(include=[np.number])
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"  特征维度: {X.shape[1]}")
        print(f"  类别: {self.label_encoder.classes_}")
        unique, counts = np.unique(y_encoded, return_counts=True)
        print(f"  原始类别分布: {dict(zip(self.label_encoder.classes_, counts))}")
        
        return X.values, y_encoded
    
    def _convert_format(self, X: np.ndarray, model_type: str) -> np.ndarray:
        """格式转换"""
        print(f"[4/4] 转换为 {model_type} 格式...")
        
        if model_type == 'resnet':
            return self._to_resnet(X)
        elif model_type == 'lstm':
            return self._to_lstm(X)
        elif model_type == 'cnn_rnn':
            return self._to_cnn_rnn(X)
      
    def _to_resnet(self, X: np.ndarray) -> np.ndarray:
        """改进版：全局标准化 + 智能填充"""
        n_samples = X.shape[0]
        target = IMG_SIZE * IMG_SIZE  # 784
        
        X = X.astype(np.float32)
        
        # ========== 关键修改1：简单特征选择（去除无效列）==========
        if self.feature_indices is None:
            # 首次调用：计算有效特征（非零方差）
            variances = np.var(X, axis=0)
            self.feature_indices = np.where(variances > 1e-10)[0]  # 去除常数列
            print(f"  特征选择: {X.shape[1]} -> {len(self.feature_indices)} 维 (去除方差为0的特征)")
        
        X = X[:, self.feature_indices]
        
        # ========== 关键修改2：全局标准化（替代Min-Max）==========
        if self.feature_means is None:
            # 训练集：计算全局统计量
            self.feature_means = X.mean(axis=0)
            self.feature_stds = X.std(axis=0) + 1e-8
        
        # Z-score标准化（均值为0，标准差为1），比Min-Max更鲁棒
        X_norm = (X - self.feature_means) / self.feature_stds
        
        # 截断异常值（CICIDS中有极大值如1e9）
        X_norm = np.clip(X_norm, -5, 5)  # 限制在±5个标准差内
        
        # ========== 保持原有 reshape 逻辑 ==========
        if X_norm.shape[1] < target:
            padding = np.zeros((n_samples, target - X_norm.shape[1]), dtype=np.float32)
            X_pad = np.concatenate([X_norm, padding], axis=1)
        else:
            X_pad = X_norm[:, :target]
        
        X_img = X_pad.reshape(n_samples, 1, IMG_SIZE, IMG_SIZE)
        print(f"  输出: {X_img.shape}, 值范围: [{X_img.min():.2f}, {X_img.max():.2f}]")
        return X_img
    
        
    def _to_lstm(self, X: np.ndarray) -> np.ndarray:
        """转为时序 (N, 50, 2)"""
        n_samples, n_feats = X.shape
        
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        use_feats = min(100, n_feats)
        X_trim = X_norm[:, :use_feats]
        
        if use_feats < 100:
            pad_width = ((0, 0), (0, 100 - use_feats))
            X_trim = np.pad(X_trim, pad_width, mode='constant')
        
        X_seq = X_trim.reshape(n_samples, SEQ_LEN, 2)
        print(f"  输出形状: {X_seq.shape}")
        return X_seq.astype(np.float32)
    
    def _to_cnn_rnn(self, X: np.ndarray) -> np.ndarray:
        """转为字节矩阵 (N, 50, 256) - 假设输入已分层采样"""
        n_samples = X.shape[0]
        target = SEQ_LEN * PACKET_BYTES
        
        X_min = X.min()
        X_max = X.max()
        X_bytes = ((X - X_min) / (X_max - X_min + 1e-8) * 255).astype(np.uint8)
        
        if X.shape[1] < target:
            X_bytes = np.pad(X_bytes, ((0,0),(0,target-X.shape[1])), mode='constant')
        else:
            X_bytes = X_bytes[:, :target]
        
        X_mat = X_bytes.reshape(n_samples, SEQ_LEN, PACKET_BYTES).astype(np.float16) / 255.0
        print(f"  输出形状: {X_mat.shape}, 内存: ~{X_mat.nbytes/1024/1024:.0f}MB")
        return X_mat


    def _save(self, X_train, X_test, y_train, y_test, model_type):
        """保存时加入预处理参数"""
        output_file = OUTPUT_DIR / f"CIC_{model_type}_data.npz"
        
        save_dict = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'classes': self.label_encoder.classes_
        }
        
        # 保存归一化参数（resnet模式需要）
        if model_type == 'resnet' and self.feature_means is not None:
            save_dict['feature_means'] = self.feature_means
            save_dict['feature_stds'] = self.feature_stds
            save_dict['feature_indices'] = self.feature_indices
            
        np.savez(output_file, **save_dict)
        print(f"[✓] 已保存: {output_file}")



def main():
    processor = CICPreprocessor()
    
    for model in ['resnet', 'lstm', 'cnn_rnn']:
        try:
            processor.process(model)
            print()
        except Exception as e:
            print(f"[✗] {model} 失败: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()