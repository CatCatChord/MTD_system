<template>
    <header class="navbar">
      <div class="brand">
        <el-icon :size="24" class="logo-icon"><Platform /></el-icon>
        <span class="title">网络威胁感知系统 <small>v2.0 Pro</small></span>
      </div>
  
      <div class="controls-area">
        <el-button-group class="mode-switch">
          <el-button
            type="primary"
            :plain="viewMode !== 'realtime'"
            @click="$emit('switch-view', 'realtime')"
          >
            <el-icon style="margin-right:5px"><Monitor /></el-icon> 实时监控
          </el-button>
          <el-button
            type="success"
            :plain="viewMode !== 'benchmark'"
            @click="$emit('switch-view', 'benchmark')"
          >
            <el-icon style="margin-right:5px"><Cpu /></el-icon> 模型自检
          </el-button>
        </el-button-group>
  
        <div v-if="viewMode === 'realtime'" class="model-control">
          <span class="label">防御引擎:</span>
          <el-tag effect="dark" type="warning" class="model-tag">
            {{ currentModelName || 'Loading...' }}
          </el-tag>
          <el-dropdown trigger="click" @command="$emit('switch-model', $event)">
            <el-button type="primary" size="small" class="switch-btn">
              切换 <el-icon class="el-icon--right"><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="resnet">ResNet (空间特征)</el-dropdown-item>
                <el-dropdown-item command="lstm">LSTM (时序特征)</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </div>
    </header>
  </template>
  
  <script setup>
  import { Platform, ArrowDown, Monitor, Cpu } from '@element-plus/icons-vue'
  
  defineProps({
    viewMode: { type: String, required: true },
    currentModelName: { type: String, default: 'Loading...' }
  })
  defineEmits(['switch-view', 'switch-model'])
  </script>
  
  <style scoped>
  .navbar {
    height: 72px;
    background: rgba(21, 30, 50, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-color);
    display: flex; justify-content: space-between; align-items: center;
    padding: 0 32px;
  }
  .brand { display: flex; align-items: center; gap: 12px; font-size: 24px; font-weight: 700; color: var(--accent-blue); letter-spacing: 1px; }
  .brand small { font-size: 12px; color: var(--text-secondary); opacity: 0.7; margin-left: 5px; }
  .controls-area { display: flex; gap: 24px; align-items: center; }
  .model-control { display: flex; align-items: center; gap: 12px; border-left: 1px solid var(--border-color); padding-left: 24px; }
  .model-control .label { font-size: 14px; color: var(--text-secondary); }
  </style>