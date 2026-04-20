# --- START OF FILE main.py ---

import os
import time
import tempfile
import threading
import sqlite3
import json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import accuracy_score
from scapy.all import sniff, PcapReader, IP, TCP, UDP, Raw

# 引入 HuggingFace 库用于加载 ET-BERT
from transformers import BertConfig, BertForSequenceClassification 

# 引入你的模型架构 (确保 models.py 已经就绪)
from models import (
    ResNetTraffic64, BiLSTMTrafficUSTC, CNNRNNTrafficUSTC,
    ResNet28, BiLSTMCIC, CNNRNNCIC
)

# ================= 1. 全局配置 =================
CURRENT_MODEL_NAME = "ResNet"
CURRENT_MODEL_KEY = "resnet"   # resnet / lstm / cnnrnn / etbert
CURRENT_DATASET = "ustc"
IS_SNIFFING = True
MAX_CACHE = 50
packet_queue = deque(maxlen=MAX_CACHE)
global_stats = {
    "total_packets": 0,
    "malicious_count": 0,
    "start_time": time.strftime("%H:%M:%S")
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实时流缓存
flow_cache = {}
flow_cache_lock = threading.Lock()

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), "detection_system.db")


# ================= 2. 数据库模块 =================

def init_db():
    """初始化 SQLite 数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            flow_key TEXT,
            model_name TEXT,
            dataset_type TEXT,
            prediction TEXT,
            confidence REAL,
            packets_count INTEGER,
            bytes_total INTEGER,
            is_realtime INTEGER DEFAULT 0,
            pcap_filename TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE,
            dataset_type TEXT,
            total_inferences INTEGER DEFAULT 0,
            malicious_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            last_updated TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f">>> [系统] 数据库初始化完成: {DB_PATH}")


def save_detection_record(timestamp, source, flow_key, model_name, dataset_type,
                          prediction, confidence, packets_count, bytes_total,
                          is_realtime=0, pcap_filename=None):
    """保存检测记录到数据库"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detection_records 
            (timestamp, source, flow_key, model_name, dataset_type, prediction, 
             confidence, packets_count, bytes_total, is_realtime, pcap_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, source, flow_key, model_name, dataset_type, prediction,
              confidence, packets_count, bytes_total, is_realtime, pcap_filename))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f">>> [DB错误] 保存记录失败: {e}")


def update_model_performance(model_name, dataset_type, confidence, is_malicious):
    """更新模型性能统计"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT total_inferences, malicious_count, avg_confidence 
            FROM model_performance WHERE model_name = ?
        ''', (model_name,))
        row = cursor.fetchone()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if row:
            total, mal, avg_conf = row
            total += 1
            mal += 1 if is_malicious else 0
            new_avg = (avg_conf * (total - 1) + confidence) / total
            cursor.execute('''
                UPDATE model_performance 
                SET total_inferences=?, malicious_count=?, avg_confidence=?, 
                    dataset_type=?, last_updated=?
                WHERE model_name=?
            ''', (total, mal, new_avg, dataset_type, now, model_name))
        else:
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, dataset_type, total_inferences, malicious_count, avg_confidence, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (model_name, dataset_type, 1, 1 if is_malicious else 0, confidence, now))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f">>> [DB错误] 更新性能统计失败: {e}")


def query_records(limit=100, offset=0, is_realtime=None):
    """查询检测记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if is_realtime is not None:
        cursor.execute('''
            SELECT * FROM detection_records WHERE is_realtime = ?
            ORDER BY id DESC LIMIT ? OFFSET ?
        ''', (is_realtime, limit, offset))
    else:
        cursor.execute('''
            SELECT * FROM detection_records 
            ORDER BY id DESC LIMIT ? OFFSET ?
        ''', (limit, offset))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ================= 3. 核心特征工程 =================
SEQ_LEN = 50
PACKET_BYTES = 256
IMG_SIZE = 64
MIN_PACKETS = 5

def extract_flows_from_pcap(pcap_path, max_flows=500):
    """从 PCAP 提取流，还原训练时的五元组对齐与时间排序逻辑"""
    flows_dict = {}
    try:
        with PcapReader(pcap_path) as pcap:
            for pkt in pcap:
                if IP not in pkt: continue
                
                src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
                if TCP in pkt:
                    src_port, dst_port, proto = pkt[TCP].sport, pkt[TCP].dport, 'TCP'
                    payload = bytes(pkt[TCP].payload) if Raw in pkt else b''
                    flags = pkt[TCP].flags if hasattr(pkt[TCP], 'flags') else 0
                elif UDP in pkt:
                    src_port, dst_port, proto = pkt[UDP].sport, pkt[UDP].dport, 'UDP'
                    payload = bytes(pkt[UDP].payload) if Raw in pkt else b''
                    flags = 0
                else:
                    continue
                
                # 判断方向
                if (src_ip, src_port) < (dst_ip, dst_port):
                    key = (src_ip, dst_ip, src_port, dst_port, proto)
                    direction = 0 # 上行
                else:
                    key = (dst_ip, src_ip, dst_port, src_port, proto)
                    direction = 1 # 下行
                
                if key not in flows_dict:
                    if len(flows_dict) >= max_flows: break
                    flows_dict[key] =[]
                    
                flows_dict[key].append({
                    'timestamp': float(pkt.time),
                    'size': len(pkt),
                    'payload': payload,
                    'direction': direction,
                    'flags': int(flags)
                })
    except Exception as e:
        print(f"解析 PCAP 失败: {e}")

    # 整合有效流并计算 IAT
    valid_flows =[]
    for key, pkts in flows_dict.items():
        if len(pkts) < MIN_PACKETS: continue
        
        pkts.sort(key=lambda x: x['timestamp'])
        pkts = pkts[:SEQ_LEN]
        
        for i in range(1, len(pkts)):
            pkts[i]['iat'] = pkts[i]['timestamp'] - pkts[i-1]['timestamp']
        pkts[0]['iat'] = 0.0
        
        valid_flows.append(pkts)
        
    return valid_flows

def prep_resnet(pkts, dataset_type="ustc"):
    """转换: USTC -> [3, 64, 64] float32; CIC -> [1, 28, 28] float32"""
    if dataset_type == "cic":
        target_len = 28 * 28  # 784
        all_bytes = b''.join([p['payload'] for p in pkts])
        if len(all_bytes) < target_len:
            all_bytes += b'\x00' * (target_len - len(all_bytes))
        else:
            all_bytes = all_bytes[:target_len]
        img = np.frombuffer(all_bytes, dtype=np.uint8).reshape(28, 28).astype(np.float32)
        img = (img - 128.0) / 128.0
        img = img[np.newaxis, :, :]  # [1, 28, 28]
        return img
    else:
        # USTC 默认
        all_bytes = b''.join([p['payload'] for p in pkts])
        target_len = IMG_SIZE * IMG_SIZE # 4096
        
        if len(all_bytes) < target_len:
            all_bytes += b'\x00' * (target_len - len(all_bytes))
        else:
            all_bytes = all_bytes[:target_len]
            
        img = np.frombuffer(all_bytes, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE).astype(np.float32)
        img = (img - 128.0) / 128.0
        img = np.stack([img, img, img], axis=0)
        return img

def prep_lstm(pkts, dataset_type="ustc"):
    """转换: USTC -> [50, 15] float32; CIC -> [50, 2] float32"""
    sizes = np.array([p['size'] for p in pkts], dtype=np.float32)
    iats = np.array([p['iat'] for p in pkts], dtype=np.float32)
    dirs = np.array([p['direction'] for p in pkts], dtype=np.float32)
    
    if len(sizes) < SEQ_LEN:
        sizes = np.pad(sizes, (0, SEQ_LEN - len(sizes)), mode='constant')
        iats = np.pad(iats, (0, SEQ_LEN - len(iats)), mode='constant', constant_values=0)
        dirs = np.pad(dirs, (0, SEQ_LEN - len(dirs)), mode='constant', constant_values=-1)
        
    if dataset_type == "cic":
        # CIC BiLSTM 期望 [50, 2]
        seq_features = np.column_stack([sizes, iats])  # [SEQ_LEN, 2]
        # 简单 z-score 标准化
        means = seq_features.mean(axis=0)
        stds = seq_features.std(axis=0) + 1e-8
        seq_features = (seq_features - means) / stds
        return seq_features.astype(np.float32)
    
    # USTC 默认: [50, 15]
    if len(pkts) > 0:
        size_mean, size_std = np.mean(sizes[:len(pkts)]), np.std(sizes[:len(pkts)]) if len(pkts)>1 else 0
        size_min, size_max = np.min(sizes[:len(pkts)]), np.max(sizes[:len(pkts)])
        
        valid_iats = iats[1:len(pkts)]
        iat_mean, iat_std = np.mean(valid_iats) if len(valid_iats)>0 else 0, np.std(valid_iats) if len(valid_iats)>1 else 0
        iat_max = np.max(valid_iats) if len(valid_iats)>0 else 0
        
        dir_0_ratio = np.sum(dirs[:len(pkts)] == 0) / len(pkts)
        dir_1_ratio = 1 - dir_0_ratio
        
        total_bytes = np.sum(sizes[:len(pkts)])
        bytes_up = np.sum(sizes[:len(pkts)] * (dirs[:len(pkts)] == 0))
        bytes_down = total_bytes - bytes_up
        
        duration = np.sum(iats[:len(pkts)])
        pkt_count = len(pkts) / SEQ_LEN
        
        stat_features = np.array([
            size_mean, size_std, size_min, size_max, iat_mean, iat_std, iat_max,
            dir_0_ratio, dir_1_ratio, bytes_up/1000.0, bytes_down/1000.0, duration, pkt_count
        ])
    else:
        stat_features = np.zeros(12) 
        
    seq_features = np.column_stack([sizes, iats, dirs]) 
    stat_repeated = np.tile(stat_features, (SEQ_LEN, 1)) 
    
    return np.concatenate([seq_features, stat_repeated], axis=1) 

def prep_cnnrnn(pkts, dataset_type="ustc"):
    """转换: USTC -> [50, 256] int64; CIC -> [50, 256] float32"""
    matrix = np.zeros((SEQ_LEN, PACKET_BYTES), dtype=np.int64)
    for i, p in enumerate(pkts[:SEQ_LEN]):
        payload = p['payload'][:PACKET_BYTES]
        if len(payload) < PACKET_BYTES:
            payload += b'\x00' * (PACKET_BYTES - len(payload))
        matrix[i] = np.array([b for b in payload], dtype=np.int64)
    
    if dataset_type == "cic":
        # CIC CNN+RNN 期望 float32，数值范围 [0, 1]
        return matrix.astype(np.float32) / 255.0
    else:
        # USTC 期望 int64 (用于 nn.Embedding)
        return matrix

def prep_etbert(pkts, max_len=512):
    """转换: -> dict (包含 input_ids 和 attention_mask)"""
    all_bytes = b''.join([p['payload'] for p in pkts])
    byte_seq = list(all_bytes[:max_len - 2])
    
    # 词表映射 0~255 -> 5~260
    input_ids = [2] +[b + 5 for b in byte_seq] + [3]  # [CLS] + Bytes + [SEP]
    pad_len = max_len - len(input_ids)
    
    attention_mask =[1] * len(input_ids) + [0] * pad_len
    input_ids = input_ids + [0] * pad_len
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }



# ================= 4. 动态加载模型 =================
ai_model = None

def load_model(model_name="resnet", dataset_type="ustc"):
    global ai_model, CURRENT_MODEL_NAME, CURRENT_MODEL_KEY, CURRENT_DATASET
    print(f">>> [系统] 正在加载 {dataset_type.upper()} 数据集的引擎: {model_name} ...")
    
    # 动态构建存放模型权重的路径
    base_dir = f"../deployed_models/{dataset_type}model/"
    CURRENT_DATASET = dataset_type
    CURRENT_MODEL_KEY = model_name
    
    try:
        # ---- 传统模型分支 ----
        if model_name in ["resnet", "lstm", "cnnrnn"]:
            checkpoint = None
            num_classes = 2  # 默认二分类
            
            if dataset_type == "cic":
                if model_name == "resnet":
                    path = os.path.join(base_dir, "best_resnet.pth")
                    CURRENT_MODEL_NAME = "ResNet28 (CIC)"
                elif model_name == "lstm":
                    path = os.path.join(base_dir, "best_lstm.pth")
                    CURRENT_MODEL_NAME = "BiLSTM (CIC)"
                elif model_name == "cnnrnn":
                    path = os.path.join(base_dir, "best_cnn_rnn.pth")
                    CURRENT_MODEL_NAME = "CNN+RNN (CIC)"
            else:  # ustc
                if model_name == "resnet":
                    path = os.path.join(base_dir, "best_resnet.pth")
                    CURRENT_MODEL_NAME = "ResNet64 (USTC)"
                elif model_name == "lstm":
                    path = os.path.join(base_dir, "best_lstm.pth")
                    CURRENT_MODEL_NAME = "BiLSTM (USTC)"
                elif model_name == "cnnrnn":
                    path = os.path.join(base_dir, "best_cnn_rnn.pth")
                    CURRENT_MODEL_NAME = "CNN+RNN (USTC)"
            
            if not os.path.exists(path):
                print(f">>> [错误] 权重文件缺失: {path}")
                return False
                
            raw_state = torch.load(path, map_location=DEVICE)
            if isinstance(raw_state, dict) and 'model_state_dict' in raw_state:
                checkpoint = raw_state
                state_dict = checkpoint['model_state_dict']
                num_classes = checkpoint.get('num_classes', num_classes)
            else:
                state_dict = raw_state
            
            # 根据数据集和类别数构建正确模型
            if dataset_type == "cic":
                if model_name == "resnet":
                    model = ResNet28(num_classes=num_classes)
                elif model_name == "lstm":
                    model = BiLSTMCIC(num_classes=num_classes, input_size=2)
                elif model_name == "cnnrnn":
                    model = CNNRNNCIC(num_classes=num_classes)
            else:
                if model_name == "resnet":
                    model = ResNetTraffic64(num_classes=num_classes)
                elif model_name == "lstm":
                    model = BiLSTMTrafficUSTC(num_classes=num_classes, input_size=15)
                elif model_name == "cnnrnn":
                    model = CNNRNNTrafficUSTC(num_classes=num_classes)
            
            model.load_state_dict(state_dict)

        # ---- ET-BERT 专属分支 ----
        elif model_name == "etbert":
            if dataset_type == "cic":
                config_file_path = os.path.join(base_dir, "etbert_cic_config.json")
                weight_file_path = os.path.join(base_dir, "etbert_cic_model.safetensors")
            else:
                config_file_path = os.path.join(base_dir, "bert_base_config.json")
                weight_file_path = os.path.join(base_dir, "ustc_finetuned_model.bin")
            
            print(f">>> [调试] ET-BERT 配置路径: {config_file_path}")
            print(f">>> [调试] ET-BERT 权重路径: {weight_file_path}")
            
            if not os.path.exists(config_file_path):
                print(f">>>[严重错误] 找不到配置文件 {config_file_path}")
                return False
            if not os.path.exists(weight_file_path):
                print(f">>> [严重错误] 找不到权重文件 {weight_file_path}")
                return False

            # 1. 搭建模型骨架
            config = BertConfig.from_json_file(config_file_path)
            config.num_labels = 2 
            model = BertForSequenceClassification(config)
            
            # 2. 读取权重
            if weight_file_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file_path)
                except ImportError:
                    print(">>> [严重错误] 加载 .safetensors 需要安装 safetensors 库")
                    return False
            else:
                state_dict = torch.load(weight_file_path, map_location=DEVICE)
            
            # 3. 清理多卡训练产生的 'module.' 前缀
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            # 4. 注入骨架
            model.load_state_dict(new_state_dict, strict=False)
            CURRENT_MODEL_NAME = f"ET-BERT ({dataset_type.upper()})"
            
        else:
            return False

        # 将模型送入设备并设置为验证模式
        model.to(DEVICE)
        model.eval()
        ai_model = model
        print(f">>> [系统] {CURRENT_MODEL_NAME} 加载成功！")
        return True
        
    except Exception as e:
        print(f">>> [错误] 模型加载异常: {e}")
        import traceback
        traceback.print_exc()
        return False


# ================= 5. 可解释性模块 =================

def extract_explainability(pkts, model_name, dataset_type):
    """
    提取模型可解释性信息
    返回: {
        "attention_weights": [...]   # LSTM/CNN+RNN 的注意力权重 (长度 SEQ_LEN)
        "feature_map": [[...], ...]  # ResNet 最后一层卷积特征图 (H x W)
        "top_tokens": [...]          # ET-BERT 高权重 token 位置（简化版）
    }
    """
    result = {"attention_weights": None, "feature_map": None, "top_tokens": None}
    
    if ai_model is None:
        return result
    
    try:
        with torch.enable_grad():
            if model_name == "resnet":
                x = prep_resnet(pkts, dataset_type)
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                # 逐层前向到 layer4，提取特征图
                x_out = ai_model.conv1(x_tensor)
                x_out = ai_model.bn1(x_out)
                x_out = ai_model.relu(x_out)
                x_out = ai_model.maxpool(x_out)
                x_out = ai_model.layer1(x_out)
                x_out = ai_model.layer2(x_out)
                x_out = ai_model.layer3(x_out)
                feature_map = ai_model.layer4(x_out)  # [1, 512, H, W]
                # 取平均激活图
                feature_map = feature_map.squeeze(0).mean(dim=0).detach().cpu().numpy()
                # 归一化到 0-1
                fm_min, fm_max = feature_map.min(), feature_map.max()
                if fm_max > fm_min:
                    feature_map = (feature_map - fm_min) / (fm_max - fm_min)
                result["feature_map"] = feature_map.tolist()
            
            elif model_name == "lstm":
                x = prep_lstm(pkts, dataset_type)
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lstm_out, _ = ai_model.lstm(x_tensor)
                attn_scores = ai_model.attention(lstm_out).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1)
                result["attention_weights"] = attn_weights.squeeze(0).detach().cpu().numpy().tolist()
            
            elif model_name == "cnnrnn":
                x = prep_cnnrnn(pkts, dataset_type)
                if dataset_type == "cic" or model_name in ["resnet", "lstm"]:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                else:
                    x_tensor = torch.tensor(x, dtype=torch.int64).unsqueeze(0).to(DEVICE)
                
                # CNN 部分
                x_out = x_tensor.permute(0, 2, 1)
                x_out = ai_model.cnn_down(x_out)
                x_out = x_out.permute(0, 2, 1)
                lstm_out, _ = ai_model.lstm(x_out)
                lstm_out = ai_model.layer_norm(lstm_out)
                attn_scores = ai_model.attention(lstm_out).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1)
                result["attention_weights"] = attn_weights.squeeze(0).detach().cpu().numpy().tolist()
            
            elif model_name == "etbert":
                x = prep_etbert(pkts)
                input_ids = torch.tensor([x["input_ids"]], dtype=torch.long).to(DEVICE)
                attention_mask = torch.tensor([x["attention_mask"]], dtype=torch.long).to(DEVICE)
                # 简化：返回 input_ids 中对应 payload 非零区域的 mask 作为可解释性提示
                tokens = x["input_ids"]
                # 排除 [CLS],[SEP],padding 后取前 50 个高亮位置
                top_idx = [i for i, t in enumerate(tokens) if t not in [0, 2, 3]][:50]
                result["top_tokens"] = top_idx
    except Exception as e:
        print(f">>> [可解释性] 提取失败: {e}")
    
    return result


# ================= 6. 实时嗅探与推理 =================

def infer_flow(flow_key, pkts):
    """对单个流执行AI推理（在独立线程中运行）"""
    global ai_model, CURRENT_MODEL_NAME, CURRENT_DATASET, CURRENT_MODEL_KEY
    
    if ai_model is None:
        return
    
    try:
        # 计算IAT
        pkts.sort(key=lambda x: x['timestamp'])
        for i in range(1, len(pkts)):
            pkts[i]['iat'] = pkts[i]['timestamp'] - pkts[i-1]['timestamp']
        pkts[0]['iat'] = 0.0
        
        model_key = CURRENT_MODEL_KEY
        dataset_type = CURRENT_DATASET
        
        # 预处理
        if model_key == "resnet":
            x = prep_resnet(pkts, dataset_type)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        elif model_key == "lstm":
            x = prep_lstm(pkts, dataset_type)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        elif model_key == "cnnrnn":
            x = prep_cnnrnn(pkts, dataset_type)
            if dataset_type == "cic":
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            else:
                x_tensor = torch.tensor(x, dtype=torch.int64).unsqueeze(0).to(DEVICE)
        elif model_key == "etbert":
            x = prep_etbert(pkts)
            input_ids = torch.tensor([x["input_ids"]], dtype=torch.long).to(DEVICE)
            attention_mask = torch.tensor([x["attention_mask"]], dtype=torch.long).to(DEVICE)
            x_tensor = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return
        
        with torch.no_grad():
            if model_key == "etbert":
                outputs = ai_model(**x_tensor).logits
            else:
                outputs = ai_model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = torch.max(probs).item()
        
        pred_label = "Malicious" if pred != 0 else "Normal"
        
        # 更新恶意计数
        if pred != 0:
            global_stats["malicious_count"] += 1
        
        # 记录到数据库
        src, dst, sport, dport, proto = flow_key
        flow_desc = f"{src}:{sport}->{dst}:{dport}/{proto}"
        save_detection_record(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="realtime_sniffer",
            flow_key=flow_desc,
            model_name=CURRENT_MODEL_NAME,
            dataset_type=dataset_type,
            prediction=pred_label,
            confidence=conf,
            packets_count=len(pkts),
            bytes_total=sum(p['size'] for p in pkts),
            is_realtime=1
        )
        update_model_performance(CURRENT_MODEL_NAME, dataset_type, conf, pred != 0)
        
        # 加入日志队列
        packet_queue.append({
            "id": global_stats["total_packets"],
            "timestamp": time.strftime("%H:%M:%S"),
            "src": src,
            "dst": dst,
            "protocol": proto,
            "length": sum(p['size'] for p in pkts),
            "prediction": pred_label,
            "confidence": conf
        })
        
        print(f">>> [实时推理] {flow_desc} -> {pred_label} (conf={conf:.3f})")
        
    except Exception as e:
        print(f">>> [推理异常] {e}")
        import traceback
        traceback.print_exc()


def packet_callback(packet):
    """实时捕获数据包，缓存到flow池中，满足条件后触发AI推理"""
    if not IS_SNIFFING:
        return
    
    if IP not in packet:
        return
    
    try:
        src_ip, dst_ip = packet[IP].src, packet[IP].dst
        if TCP in packet:
            src_port, dst_port, proto = packet[TCP].sport, packet[TCP].dport, 'TCP'
            payload = bytes(packet[TCP].payload) if Raw in packet else b''
            flags = int(packet[TCP].flags) if hasattr(packet[TCP], 'flags') else 0
        elif UDP in packet:
            src_port, dst_port, proto = packet[UDP].sport, packet[UDP].dport, 'UDP'
            payload = bytes(packet[UDP].payload) if Raw in packet else b''
            flags = 0
        else:
            return
        
        # 标准化五元组key
        if (src_ip, src_port) < (dst_ip, dst_port):
            key = (src_ip, dst_ip, src_port, dst_port, proto)
            direction = 0
        else:
            key = (dst_ip, src_ip, dst_port, src_port, proto)
            direction = 1
        
        pkt_info = {
            'timestamp': time.time(),
            'size': len(packet),
            'payload': payload,
            'direction': direction,
            'flags': flags
        }
        
        trigger = False
        with flow_cache_lock:
            if key not in flow_cache:
                flow_cache[key] = {
                    'packets': [],
                    'last_active': time.time(),
                    'inferred': False
                }
            flow_cache[key]['packets'].append(pkt_info)
            flow_cache[key]['last_active'] = time.time()
            pkts = flow_cache[key]['packets']
            
            # 达到最小包数且未推理过，触发推理
            if len(pkts) >= MIN_PACKETS and not flow_cache[key]['inferred']:
                flow_cache[key]['inferred'] = True
                trigger = True
                flow_copy = pkts[:SEQ_LEN].copy()
        
        if trigger:
            # 异步触发推理（避免阻塞 sniff）
            t = threading.Thread(target=infer_flow, args=(key, flow_copy), daemon=True)
            t.start()
        
        # 更新全局统计与日志
        global_stats["total_packets"] += 1
        proto_str = "TCP" if proto == 'TCP' else "UDP"
        
        packet_queue.append({
            "id": global_stats["total_packets"],
            "timestamp": time.strftime("%H:%M:%S"),
            "src": src_ip,
            "dst": dst_ip,
            "protocol": proto_str,
            "length": len(packet),
            "prediction": "Analyzing" if not trigger else "Pending",
            "confidence": 0.0
        })
        
    except Exception as e:
        print(f">>> [嗅探回调异常] {e}")


def flow_cleanup_worker():
    """定期清理过期流缓存"""
    while True:
        time.sleep(30)
        now = time.time()
        with flow_cache_lock:
            expired = [k for k, v in flow_cache.items() if now - v['last_active'] > 60]
            for k in expired:
                del flow_cache[k]
            if expired:
                print(f">>> [系统] 清理 {len(expired)} 条过期流缓存")


def start_sniffer():
    print(">>> [系统] 嗅探线程启动...")
    while True:
        if IS_SNIFFING:
            try:
                sniff(prn=packet_callback, count=1, store=False, timeout=1)
            except Exception as e:
                time.sleep(1)
        else:
            time.sleep(1)


# ================= 7. 应用生命周期与App创建 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model("resnet", "ustc")
    threading.Thread(target=start_sniffer, daemon=True).start()
    threading.Thread(target=flow_cleanup_worker, daemon=True).start()
    yield

app = FastAPI(title="网络威胁感知系统后端 v3.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= 8. API 路由 =================

# ---------- 离线端到端测试 ----------
@app.post("/api/upload_and_test")
async def upload_pcap_and_test(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    dataset_type: str = Form("ustc"),
    ground_truth: int = Form(-1)
):
    success = load_model(model_name, dataset_type)
    if not success or not ai_model:
        return {"status": "error", "message": f"模型加载失败: 找不到 {dataset_type} 数据集的 {model_name} 权重"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        print(f">>>[PCAP分析] {file.filename}, 引擎: {model_name}, 数据集: {dataset_type}")
        
        flows = extract_flows_from_pcap(tmp_path, max_flows=500)
        if not flows:
            return {"status": "error", "message": "PCAP 中无有效TCP/UDP会话流"}

        X_batch = []
        flow_details =[]
        
        for flow in flows:
            # 记录流基础信息
            first_pkt = flow[0]
            flow_info = {
                "timestamp": first_pkt['timestamp'],
                "packets_count": len(flow),
                "bytes_total": sum(p['size'] for p in flow)
            }
            flow_details.append(flow_info)

            # 预处理（根据数据集类型选择对应格式）
            if model_name == "resnet": X_batch.append(prep_resnet(flow, dataset_type))
            elif model_name == "lstm": X_batch.append(prep_lstm(flow, dataset_type))
            elif model_name == "cnnrnn": X_batch.append(prep_cnnrnn(flow, dataset_type))
            elif model_name == "etbert": X_batch.append(prep_etbert(flow))
        
        # 张量转换与推理（分批进行，防止 ET-BERT 等大模型 OOM）
        BATCH_INFER_SIZE = 8
        all_outputs = []
        with torch.no_grad():
            if model_name == "etbert":
                for i in range(0, len(X_batch), BATCH_INFER_SIZE):
                    batch_slice = X_batch[i:i+BATCH_INFER_SIZE]
                    input_ids = torch.tensor([x["input_ids"] for x in batch_slice], dtype=torch.long).to(DEVICE)
                    attention_mask = torch.tensor([x["attention_mask"] for x in batch_slice], dtype=torch.long).to(DEVICE)
                    batch_outputs = ai_model(input_ids=input_ids, attention_mask=attention_mask).logits
                    all_outputs.append(batch_outputs)
            else:
                # 根据数据集+模型决定数据类型
                if dataset_type == "cic" or model_name in ["resnet", "lstm"]:
                    X_array = np.array(X_batch)
                else:
                    X_array = np.array(X_batch)
                
                for i in range(0, len(X_batch), BATCH_INFER_SIZE):
                    batch_slice = X_array[i:i+BATCH_INFER_SIZE]
                    if dataset_type == "cic" or model_name in ["resnet", "lstm"]:
                        X_tensor = torch.tensor(batch_slice, dtype=torch.float32).to(DEVICE)
                    else:
                        X_tensor = torch.tensor(batch_slice, dtype=torch.int64).to(DEVICE)
                    batch_outputs = ai_model(X_tensor)
                    all_outputs.append(batch_outputs)
            
            outputs = torch.cat(all_outputs, dim=0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = torch.max(probs, dim=1)[0].cpu().numpy()

        correct = 0
        malicious_found = 0
        
        # 整合分析结果并持久化
        for i in range(len(preds)):
            pred_label = int(preds[i])
            flow_details[i]["prediction"] = "Malicious" if pred_label != 0 else "Normal"
            flow_details[i]["confidence"] = float(confidences[i])
            
            if pred_label != 0: 
                malicious_found += 1
            if ground_truth != -1 and pred_label == ground_truth:
                correct += 1
            
            # 写入数据库
            save_detection_record(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                source="offline_pcap",
                flow_key=f"flow_{i}",
                model_name=CURRENT_MODEL_NAME,
                dataset_type=dataset_type,
                prediction="Malicious" if pred_label != 0 else "Normal",
                confidence=float(confidences[i]),
                packets_count=flow_details[i]["packets_count"],
                bytes_total=flow_details[i]["bytes_total"],
                is_realtime=0,
                pcap_filename=file.filename
            )
            update_model_performance(CURRENT_MODEL_NAME, dataset_type, float(confidences[i]), pred_label != 0)

        acc = (correct / len(preds)) * 100 if ground_truth != -1 else None

        return {
            "status": "success",
            "filename": file.filename,
            "total_flows": len(preds),
            "malicious_flows": malicious_found,
            "correct": correct if ground_truth != -1 else None,
            "accuracy": round(acc, 2) if acc is not None else None,
            "flow_details": flow_details
        }
    except Exception as e:
        print(f"分析出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(tmp_path)


# ---------- 可解释性 API ----------
class ExplainReq(BaseModel):
    pcap_path: str = None
    model_name: str = "resnet"
    dataset_type: str = "ustc"

@app.post("/api/explain")
async def explain_endpoint(req: ExplainReq):
    """
    可解释性分析接口：对输入的流量提取注意力权重或特征图
    """
    success = load_model(req.model_name, req.dataset_type)
    if not success or not ai_model:
        return {"status": "error", "message": "模型加载失败"}
    
    if not req.pcap_path or not os.path.exists(req.pcap_path):
        return {"status": "error", "message": "请提供有效的 PCAP 文件路径"}
    
    flows = extract_flows_from_pcap(req.pcap_path, max_flows=1)
    if not flows:
        return {"status": "error", "message": "PCAP 中无有效流"}
    
    flow = flows[0]
    explain_data = extract_explainability(flow, req.model_name, req.dataset_type)
    
    # 同时给出预测结果
    if req.model_name == "resnet":
        x = torch.tensor(prep_resnet(flow, req.dataset_type), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred = torch.argmax(torch.softmax(ai_model(x), dim=1), dim=1).item()
        conf = torch.max(torch.softmax(ai_model(x), dim=1)).item()
    elif req.model_name == "lstm":
        x = torch.tensor(prep_lstm(flow, req.dataset_type), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred = torch.argmax(torch.softmax(ai_model(x), dim=1), dim=1).item()
        conf = torch.max(torch.softmax(ai_model(x), dim=1)).item()
    elif req.model_name == "cnnrnn":
        x = prep_cnnrnn(flow, req.dataset_type)
        if req.dataset_type == "cic":
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            x = torch.tensor(x, dtype=torch.int64).unsqueeze(0).to(DEVICE)
        pred = torch.argmax(torch.softmax(ai_model(x), dim=1), dim=1).item()
        conf = torch.max(torch.softmax(ai_model(x), dim=1)).item()
    elif req.model_name == "etbert":
        x = prep_etbert(flow)
        input_ids = torch.tensor([x["input_ids"]], dtype=torch.long).to(DEVICE)
        attention_mask = torch.tensor([x["attention_mask"]], dtype=torch.long).to(DEVICE)
        pred = torch.argmax(torch.softmax(ai_model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=1), dim=1).item()
        conf = torch.max(torch.softmax(ai_model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=1)).item()
    else:
        pred, conf = 0, 0.0
    
    return {
        "status": "success",
        "model": CURRENT_MODEL_NAME,
        "prediction": "Malicious" if pred != 0 else "Normal",
        "confidence": float(conf),
        "packets_count": len(flow),
        "explainability": explain_data
    }


# ---------- 模型切换 ----------
class ModelSwitchReq(BaseModel):
    model_name: str
    dataset_type: str = "ustc"

@app.post("/api/switch_model")
def switch_model_endpoint(req: ModelSwitchReq):
    if load_model(req.model_name, req.dataset_type):
        return {"status": "success", "current_model": CURRENT_MODEL_NAME}
    return {"status": "error", "message": "模型权重加载失败"}


# ---------- Dashboard ----------
@app.get("/api/dashboard")
def get_dashboard_data():
    return {
        "stats": global_stats,
        "traffic_log": list(packet_queue),
        "current_model": CURRENT_MODEL_NAME
    }


# ---------- 检测记录查询 ----------
@app.get("/api/records")
def get_records(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), realtime: int = Query(None)):
    """查询检测记录"""
    is_rt = realtime if realtime is not None else None
    rows = query_records(limit=limit, offset=offset, is_realtime=is_rt)
    return {"status": "success", "count": len(rows), "records": rows}


# ---------- 模型性能统计 ----------
@app.get("/api/performance")
def get_performance():
    """获取模型性能统计"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM model_performance ORDER BY total_inferences DESC')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"status": "success", "models": rows}


# ================= 9. 入口 =================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18000)

# --- END OF FILE main.py ---
