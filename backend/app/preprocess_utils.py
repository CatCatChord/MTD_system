import numpy as np
import torch
from scapy.all import PcapReader, IP, TCP, UDP, Raw

# ================= 1. PCAP 提取流 =================
def extract_flows_from_pcap(pcap_path, max_flows=500):
    """从 PCAP 中提取双向流，按五元组聚合"""
    flows = {}
    try:
        with PcapReader(pcap_path) as pcap:
            for pkt in pcap:
                if IP in pkt and (TCP in pkt or UDP in pkt):
                    proto = pkt[IP].proto
                    sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
                    dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
                    
                    # 简化五元组作为 key
                    key = (pkt[IP].src, pkt[IP].dst, sport, dport, proto)
                    
                    if key not in flows:
                        if len(flows) >= max_flows: break
                        flows[key] =[]
                    
                    # 限制每个流最多 50 个包
                    if len(flows[key]) < 50:
                        flows[key].append(pkt)
    except Exception as e:
        print(f"解析 PCAP 失败: {e}")
    return list(flows.values())

# ================= 2. 四种模型的预处理逻辑 =================

def prep_resnet(flow_packets):
    """提取 784 字节载荷，转为 (1, 28, 28) 灰度图"""
    payload_limit = 784
    raw_payload = b''
    for pkt in flow_packets:
        if TCP in pkt: raw_payload += bytes(pkt[TCP].payload)
        elif UDP in pkt: raw_payload += bytes(pkt[UDP].payload)
        if len(raw_payload) >= payload_limit: break
            
    arr = np.frombuffer(raw_payload, dtype=np.uint8)
    if len(arr) < payload_limit:
        arr = np.pad(arr, (0, payload_limit - len(arr)), 'constant')
    else:
        arr = arr[:payload_limit]
        
    return arr.reshape(1, 28, 28).astype(np.float32) / 255.0

def prep_lstm(flow_packets):
    """提取 (50, 2) 的 [Length, IAT] 序列"""
    lengths, iats = [], []
    last_time = float(flow_packets[0].time)
    
    for pkt in flow_packets:
        lengths.append(len(pkt))
        curr_time = float(pkt.time)
        iats.append(curr_time - last_time)
        last_time = curr_time
        
    pad_len = 50 - len(lengths)
    if pad_len > 0:
        lengths += [0.0] * pad_len
        iats += [0.0] * pad_len
        
    # 对数归一化
    lengths = np.log1p(np.array(lengths, dtype=np.float32)) / 8.0  
    iats = np.log1p(np.array(iats, dtype=np.float32) * 1000) / 10.0 
    return np.stack([lengths, iats], axis=1)

def prep_cnnrnn(flow_packets):
    """提取 (50, 1480) 的原始字节矩阵"""
    matrix = np.zeros((50, 1480), dtype=np.float32)
    for i, pkt in enumerate(flow_packets):
        raw_bytes = bytes(pkt)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        if len(arr) > 1480: arr = arr[:1480]
        matrix[i, :len(arr)] = arr.astype(np.float32) / 255.0
    return matrix

def prep_etbert(flow_packets):
    """提取前 128 字节的 Hex Tokens 作为文本句子"""
    hex_tokens = []
    for pkt in flow_packets[:50]:
        payload = b''
        if TCP in pkt: payload = bytes(pkt[TCP].payload)[:128]
        elif UDP in pkt: payload = bytes(pkt[UDP].payload)[:128]
        hex_tokens.extend([f"{b:02x}" for b in payload])
            
    return "[CLS] " + " ".join(hex_tokens) + " [SEP]"