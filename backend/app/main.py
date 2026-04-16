# --- START OF FILE main.py ---

import os
import time
import tempfile
import threading
from collections import deque

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
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

app = FastAPI(title="网络威胁感知系统后端 v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. 核心特征工程 =================
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


# ================= 3. 动态加载模型 =================
ai_model = None

def load_model(model_name="resnet", dataset_type="ustc"):
    global ai_model, CURRENT_MODEL_NAME, CURRENT_DATASET
    print(f">>> [系统] 正在加载 {dataset_type.upper()} 数据集的引擎: {model_name} ...")
    
    # 动态构建存放模型权重的路径
    base_dir = f"../deployed_models/{dataset_type}model/"
    CURRENT_DATASET = dataset_type
    
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


# ================= 4. 离线端到端测试 API =================
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
        
        # 张量转换与推理
        with torch.no_grad():
            if model_name == "etbert":
                input_ids = torch.tensor([x["input_ids"] for x in X_batch], dtype=torch.long).to(DEVICE)
                attention_mask = torch.tensor([x["attention_mask"] for x in X_batch], dtype=torch.long).to(DEVICE)
                outputs = ai_model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                # 根据数据集+模型决定数据类型
                if dataset_type == "cic" or model_name in ["resnet", "lstm"]:
                    X_tensor = torch.tensor(np.array(X_batch), dtype=torch.float32).to(DEVICE)
                else:
                    # USTC cnnrnn 使用 int64
                    X_tensor = torch.tensor(np.array(X_batch), dtype=torch.int64).to(DEVICE)
                outputs = ai_model(X_tensor)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = torch.max(probs, dim=1)[0].cpu().numpy()

        correct = 0
        malicious_found = 0
        
        # 整合分析结果
        for i in range(len(preds)):
            pred_label = int(preds[i])
            flow_details[i]["prediction"] = "Malicious" if pred_label != 0 else "Normal"
            flow_details[i]["confidence"] = float(confidences[i])
            
            if pred_label != 0: 
                malicious_found += 1
            if ground_truth != -1 and pred_label == ground_truth:
                correct += 1

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


# ================= 5. 其他基础控制 API =================
class ModelSwitchReq(BaseModel):
    model_name: str
    dataset_type: str = "ustc" # 默认值，适配前端基础切换

@app.post("/api/switch_model")
def switch_model_endpoint(req: ModelSwitchReq):
    if load_model(req.model_name, req.dataset_type):
        return {"status": "success", "current_model": CURRENT_MODEL_NAME}
    return {"status": "error", "message": "模型权重加载失败"}

@app.get("/api/dashboard")
def get_dashboard_data():
    return {
        "stats": global_stats,
        "traffic_log": list(packet_queue),
        "current_model": CURRENT_MODEL_NAME
    }

# ================= 6. 实时嗅探 =================
def packet_callback(packet):
    """实时监控界面模拟日志生成"""
    if IP in packet:
        global_stats["total_packets"] += 1
        proto_num = packet[IP].proto
        protocol = "TCP" if proto_num == 6 else "UDP" if proto_num == 17 else "Other"
        
        packet_queue.append({
            "id": global_stats["total_packets"],
            "timestamp": time.strftime("%H:%M:%S"),
            "src": packet[IP].src,
            "dst": packet[IP].dst,
            "protocol": protocol,
            "length": len(packet),
            "prediction": "Normal", 
            "confidence": 0.99
        })

def start_sniffer():
    print(">>> [系统] 嗅探线程启动...")
    while True:
        if IS_SNIFFING:
            try: sniff(prn=packet_callback, count=1, store=False)
            except: time.sleep(1)
        else: time.sleep(1)

@app.on_event("startup")
async def startup_event():
    load_model("resnet", "ustc")
    threading.Thread(target=start_sniffer, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18000)

# --- END OF FILE main.py ---
