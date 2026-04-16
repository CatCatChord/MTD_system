<template>
    <div class="benchmark-view">
      <div class="benchmark-controls">
        <el-select
          :model-value="selectedDataset"
          @update:model-value="$emit('update:selectedDataset', $event)"
          placeholder="选择测试数据集"
          style="width: 260px"
          :disabled="loading"
        >
          <el-option
            v-for="item in datasetOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-button
          type="primary"
          @click="$emit('run')"
          :loading="loading"
        >
          {{ result ? '重新测试' : '开始测试' }}
        </el-button>
        <span v-if="loading" class="benchmark-hint">
          <el-icon class="is-loading"><Loading /></el-icon> 正在推理样本...
        </span>
      </div>
  
      <div v-if="result" class="result-box">
        <div class="result-header">
          <el-icon color="#67c23a" :size="24"><CircleCheckFilled /></el-icon>
          <span>测试完成 | 数据集: {{ selectedDataset }} | 样本数: {{ result.metrics.samples }}</span>
        </div>
        <div class="metrics-grid">
          <div class="metric-item">
            <div class="m-label">准确率 (Accuracy)</div>
            <div class="m-value text-green">{{ result.metrics.accuracy }}%</div>
          </div>
          <div class="metric-item">
            <div class="m-label">精确率 (Precision)</div>
            <div class="m-value text-blue">{{ result.metrics.precision }}%</div>
          </div>
          <div class="metric-item">
            <div class="m-label">召回率 (Recall)</div>
            <div class="m-value text-yellow">{{ result.metrics.recall }}%</div>
          </div>
          <div class="metric-item">
            <div class="m-label">F1 分数</div>
            <div class="m-value text-purple">{{ result.metrics.f1 }}%</div>
          </div>
        </div>
        <div class="cm-section">
          <p class="cm-title">混淆矩阵 (Confusion Matrix)</p>
          <div class="cm-grid">
            <div class="cm-cell bg-dark">
              <span class="cm-label">预测正常</span>
              <span class="cm-label">预测恶意</span>
            </div>
            <div class="cm-cell bg-dark"><span class="cm-label">真实正常</span></div>
            <div class="cm-cell cm-tn">
              TN: {{ result.confusion_matrix[0][0] }}
              <small>真阴性</small>
            </div>
            <div class="cm-cell cm-fp">
              FP: {{ result.confusion_matrix[0][1] }}
              <small>误报</small>
            </div>
            <div class="cm-cell bg-dark"><span class="cm-label">真实恶意</span></div>
            <div class="cm-cell cm-fn">
              FN: {{ result.confusion_matrix[1][0] }}
              <small>漏报</small>
            </div>
            <div class="cm-cell cm-tp">
              TP: {{ result.confusion_matrix[1][1] }}
              <small>真阳性</small>
            </div>
          </div>
        </div>
      </div>
      <div v-else-if="!loading" class="benchmark-placeholder">
        <el-icon :size="50"><DataLine /></el-icon>
        <p>点击“开始测试”加载数据集并评估模型性能</p>
      </div>
    </div>
  </template>
  
  <script setup>
  import { Loading, CircleCheckFilled, DataLine } from '@element-plus/icons-vue'
  
  defineProps({
    selectedDataset: { type: String, required: true },
    loading: { type: Boolean, default: false },
    result: { type: Object, default: null },
    datasetOptions: { type: Array, required: true }
  })
  defineEmits(['update:selectedDataset', 'run'])
  </script>
  
  <style scoped>
  .benchmark-view {
    overflow-y: auto;
  }
  .benchmark-controls {
    display: flex;
    align-items: center;
    gap: 20px;
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
  }
  .benchmark-hint {
    color: var(--accent-blue);
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
  }
  .benchmark-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    background: var(--card-bg);
    border: 1px dashed var(--border-color);
    border-radius: 12px;
    color: var(--text-secondary);
    gap: 16px;
  }
  .result-box {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
  }
  .result-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
    color: var(--accent-green); font-weight: bold;
  }
  .metrics-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;
  }
  .metric-item {
    background: rgba(0,0,0,0.2); border: 1px solid var(--border-color);
    border-radius: 10px; padding: 16px; text-align: center;
    position: relative; overflow: hidden;
  }
  .metric-item::after {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.05), transparent 70%);
    opacity: 0; transition: opacity 0.3s;
  }
  .metric-item:hover::after { opacity: 1; }
  .m-label { color: var(--text-secondary); font-size: 12px; margin-bottom: 8px; }
  .m-value { font-size: 26px; font-weight: bold; }
  .text-green { color: var(--accent-green); }
  .text-blue { color: var(--accent-blue); }
  .text-yellow { color: var(--accent-yellow); }
  .text-purple { color: var(--accent-purple); }
  
  .cm-section {
    background: rgba(0,0,0,0.2); padding: 16px; border-radius: 10px; border: 1px solid var(--border-color);
  }
  .cm-title { margin: 0 0 12px 0; color: var(--text-secondary); font-size: 12px; text-transform: uppercase; }
  .cm-grid {
    display: grid; grid-template-columns: 80px 1fr 1fr; gap: 2px; text-align: center; font-size: 14px;
  }
  .cm-cell {
    padding: 12px; background: rgba(255,255,255,0.03);
    display: flex; flex-direction: column; justify-content: center; align-items: center; height: 70px;
  }
  .bg-dark { background: transparent; color: var(--text-secondary); font-size: 12px; }
  .cm-tp { background: rgba(0, 229, 176, 0.1); color: var(--accent-green); font-weight: bold; border: 1px solid var(--accent-green); }
  .cm-fp { background: rgba(255, 59, 92, 0.1); color: var(--accent-red); font-weight: bold; }
  .cm-fn { background: rgba(230, 162, 60, 0.1); color: var(--accent-yellow); font-weight: bold; }
  .cm-tn { background: rgba(41, 121, 255, 0.1); color: var(--accent-blue); font-weight: bold; }
  </style>