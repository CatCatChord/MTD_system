<template>
    <div class="stat-grid">
      <div class="cyber-card">
        <div class="card-header">
          <el-icon :size="20" color="#2979ff"><DataLine /></el-icon>
          <span class="card-title">捕获数据包</span>
          <el-tag size="small" effect="dark" type="info" style="margin-left: auto;">实时</el-tag>
        </div>
        <div class="card-num mono text-blue">{{ stats.total_packets || 0 }}</div>
        <div class="card-footer">累计总量</div>
      </div>
      <div class="cyber-card">
        <div class="card-header">
          <el-icon :size="20" color="#ff3b5c"><Warning /></el-icon>
          <span class="card-title">恶意威胁</span>
        </div>
        <div class="card-num mono text-red">{{ stats.malicious_count || 0 }}</div>
        <div class="card-footer">拦截率: {{ calcRate(stats.malicious_count, stats.total_packets) }}%</div>
      </div>
      <div class="cyber-card">
        <div class="card-header">
          <el-icon :size="20" color="#00e5b0"><Monitor /></el-icon>
          <span class="card-title">系统状态</span>
        </div>
        <div class="card-status">
          <span class="status-dot"></span> 在线监测中
        </div>
        <div class="card-footer">运行时间: {{ stats.start_time || '--:--' }}</div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { DataLine, Warning, Monitor } from '@element-plus/icons-vue'
  
  defineProps({
    stats: { type: Object, required: true },
    calcRate: { type: Function, required: true }
  })
  </script>
  
  <style scoped>
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
  }
  
  .cyber-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .cyber-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 48px rgba(0, 229, 176, 0.15);
    border-color: rgba(0, 229, 176, 0.3);
  }
  
  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }
  .card-title {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .card-num {
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .card-footer {
    font-size: 11px;
    color: var(--text-secondary);
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 8px;
    margin-top: 8px;
  }
  .status-dot {
    width: 8px; height: 8px; background: var(--accent-green); border-radius: 50%;
    display: inline-block; box-shadow: 0 0 8px var(--accent-green); margin-right: 5px;
  }
  .text-blue { color: var(--accent-blue); }
  .text-red { color: var(--accent-red); }
  </style>