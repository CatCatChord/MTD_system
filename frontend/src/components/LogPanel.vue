<template>
    <div class="cyber-panel log-panel">
      <div class="panel-header">
        <el-icon><List /></el-icon> 威胁检测日志
        <el-tag size="small" type="success" effect="dark" style="margin-left: auto;">
          最新 {{ logs.length }} 条
        </el-tag>
      </div>
      <div class="table-wrapper">
        <el-table
          v-if="logs.length > 0"
          :data="logs"
          style="width: 100%; height: 100%; background: transparent;"
          :row-class-name="tableRowClassName"
          size="small"
          row-key="id"
          height="100%"
        >
          <el-table-column prop="timestamp" label="时间" width="90" />
          <el-table-column prop="protocol" label="协议" width="80">
            <template #default="scope">
              <el-tag :type="getProtocolTagType(scope.row.protocol)" size="small" effect="dark">
                {{ scope.row.protocol }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="length" label="长度" width="80" />
          <el-table-column label="AI 判定" width="100">
            <template #default="scope">
              <span :class="scope.row.prediction === 'Malicious' ? 'tag-malicious' : 'tag-normal'">
                {{ scope.row.prediction }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="置信度">
            <template #default="scope">
              <div style="display: flex; align-items: center;">
                <el-progress
                  :percentage="Math.round(scope.row.confidence * 100)"
                  :status="scope.row.prediction === 'Malicious' ? 'exception' : 'success'"
                  :stroke-width="6"
                  :show-text="false"
                  style="width: 50px"
                />
                <span style="font-size:12px; margin-left:8px; color: #94a3b8;">
                  {{ (scope.row.confidence * 100).toFixed(0) }}%
                </span>
              </div>
            </template>
          </el-table-column>
        </el-table>
        <el-empty v-else description="暂无流量数据" :image-size="80" />
      </div>
    </div>
  </template>
  
  <script setup>
  import { List } from '@element-plus/icons-vue'
  
  defineProps({
    logs: { type: Array, default: () => [] }
  })
  
  const tableRowClassName = ({ row }) => {
    return row.prediction === 'Malicious' ? 'warning-row' : ''
  }
  
  const getProtocolTagType = (protocol) => {
    const map = { TCP: 'primary', UDP: 'success', ICMP: 'warning', HTTP: 'info' }
    return map[protocol] || ''
  }
  </script>
  
  <style scoped>
  .cyber-panel {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    height: 100%;
  }
  
  .panel-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border-color);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    background: rgba(0,0,0,0.2);
    flex-shrink: 0;
  }
  
  .table-wrapper {
    flex: 1;
    min-height: 0;
    padding: 0 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .table-wrapper .el-table {
    flex: 1;
    width: 100%;
    height: 100% !important;
  }
  .table-wrapper .el-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  
  /* 表格覆写 */
  .el-table {
    --el-table-bg-color: transparent;
    --el-table-tr-bg-color: transparent;
    --el-table-header-bg-color: rgba(0,0,0,0.3);
    --el-table-text-color: var(--text-secondary);
    --el-table-header-text-color: var(--text-primary);
    --el-table-border-color: rgba(255,255,255,0.05);
  }
  .el-table td, .el-table th {
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
  }
  .el-table--enable-row-hover .el-table__body tr:hover > td {
    background-color: rgba(255,255,255,0.03) !important;
  }
  .warning-row {
    background: linear-gradient(90deg, rgba(255, 59, 92, 0.1), transparent);
  }
  .warning-row:hover td {
    background: rgba(255, 59, 92, 0.15) !important;
  }
  .tag-malicious { color: var(--accent-red); font-weight: bold; text-shadow: 0 0 5px rgba(255,59,92,0.4); }
  .tag-normal { color: var(--accent-green); }
  </style>