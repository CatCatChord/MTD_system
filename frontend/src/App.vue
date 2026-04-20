<!-- App.vue -->
<template>
  <div class="app-container">
    <!-- ================= 顶部导航栏 ================= -->
    <header class="navbar">
      <div class="brand">
        <el-icon :size="24" class="logo-icon"><Platform /></el-icon>
        <span class="title">网络威胁感知系统 <small>v3.1 System</small></span>
      </div>
      
      <div class="controls-area">
        <!-- 1. 功能模式切换 -->
        <el-button-group class="mode-switch">
          <el-button type="primary" :plain="viewMode !== 'realtime'" @click="viewMode = 'realtime'">
            <el-icon style="margin-right:5px"><Monitor /></el-icon> 实时监控
          </el-button>
          <el-button type="warning" :plain="!showOfflineTestDialog" @click="showOfflineTestDialog = true">
            <el-icon style="margin-right:5px"><UploadFilled /></el-icon> 离线 PCAP 分析
          </el-button>
          <el-button type="info" :plain="!showRecordsDialog" @click="openRecordsDialog">
            <el-icon style="margin-right:5px"><List /></el-icon> 检测记录
          </el-button>
        </el-button-group>

        <!-- 2. 在线模型切换 -->
        <div class="model-control">
          <span class="label">当前引擎:</span>
          <el-tag effect="dark" type="warning" class="model-tag">
            {{ currentModelName || 'Loading...' }}
          </el-tag>
          <el-dropdown trigger="click" @command="handleModelSwitch">
            <el-button type="primary" size="small" class="switch-btn">
              切换 <el-icon class="el-icon--right"><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="resnet">ResNet (空间特征)</el-dropdown-item>
                <el-dropdown-item command="lstm">LSTM (时序特征)</el-dropdown-item>
                <el-dropdown-item command="cnnrnn">CNN+RNN (多模态)</el-dropdown-item>
                <el-dropdown-item command="etbert">ET-BERT (语义特征)</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </div>
    </header>

    <!-- ================= 主内容区：实时监控 ================= -->
    <div class="main-content">
      <div class="stat-grid">
        <div class="cyber-card">
          <div class="card-title">捕获数据包 (Total)</div>
          <div class="card-num text-blue">{{ stats.total_packets || 0 }}</div>
        </div>
        <div class="cyber-card">
          <div class="card-title">恶意威胁 (Malicious)</div>
          <div class="card-num text-red">{{ stats.malicious_count || 0 }}</div>
          <div class="card-sub">拦截率: {{ calcRate(stats.malicious_count, stats.total_packets) }}%</div>
        </div>
        <div class="cyber-card">
          <div class="card-title">模型统计 (Model Stats)</div>
          <div class="card-num text-green">{{ modelStats.total_inferences || 0 }}</div>
          <div class="card-sub">
            检出: {{ modelStats.malicious_count || 0 }} | 
            平均置信度: {{ (modelStats.avg_confidence * 100).toFixed(1) }}%
          </div>
        </div>
        <div class="cyber-card">
          <div class="card-title">系统状态 (Status)</div>
          <div class="card-status">
            <span class="status-dot"></span> 在线监测中
          </div>
          <div class="card-sub">运行时间: {{ stats.start_time || '--:--' }}</div>
        </div>
      </div>

      <div class="dashboard-grid">
        <div class="cyber-panel chart-panel">
          <div class="panel-header"><el-icon><TrendCharts /></el-icon> 实时流量载荷分布</div>
          <div ref="chartRef" class="chart-container"></div>
        </div>

        <div class="cyber-panel log-panel">
          <div class="panel-header"><el-icon><List /></el-icon> 威胁检测日志</div>
          <el-table :data="logs" style="width: 100%; background: transparent;" height="100%" :row-class-name="tableRowClassName" size="small" row-key="id">
            <el-table-column prop="timestamp" label="时间" width="90" />
            <el-table-column prop="protocol" label="协议" width="70" />
            <el-table-column prop="length" label="长度" width="80" />
            <el-table-column label="AI 判定" width="100">
              <template #default="scope">
                <div v-if="scope && scope.row">
                  <span :class="scope.row.prediction === 'Malicious' ? 'tag-malicious' : 'tag-normal'">
                    {{ scope.row.prediction }}
                  </span>
                </div>
              </template>
            </el-table-column>
            <el-table-column label="置信度">
              <template #default="scope">
                <div v-if="scope && scope.row" style="display: flex; align-items: center;">
                  <el-progress :percentage="Math.round(scope.row.confidence * 100)" :status="scope.row.prediction === 'Malicious' ? 'exception' : 'success'" :stroke-width="6" :show-text="false" style="width: 50px" />
                  <span style="font-size:12px; margin-left:8px; color: #94a3b8;">{{ (scope.row.confidence * 100).toFixed(0) }}%</span>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </div>

    <!-- ================= 弹窗：端到端 PCAP 离线分析 ================= -->
    <el-dialog 
      v-model="showOfflineTestDialog" 
      title="端到端离线 PCAP 分析验证" 
      width="600px" 
      custom-class="cyber-dialog"
      :close-on-click-modal="false"
    >
      <el-form label-width="120px" class="cyber-form">
        <el-form-item label="分析引擎">
          <el-select v-model="testForm.model_name" style="width: 100%;">
            <el-option label="ResNet (空间灰度图特征)" value="resnet" />
            <el-option label="LSTM (时序统计特征)" value="lstm" />
            <el-option label="CNN+RNN (多模态融合特征)" value="cnnrnn" />
            <el-option label="ET-BERT (字节语义特征)" value="etbert" />
          </el-select>
        </el-form-item>

        <!-- 新增：数据集权重选择 -->
        <el-form-item label="权重配置">
          <el-radio-group v-model="testForm.dataset_type">
            <el-radio label="ustc">USTC-TFC2016 权重</el-radio>
            <el-radio label="cic">CIC-IDS2017 权重</el-radio>
          </el-radio-group>
        </el-form-item>

        <!-- 修改：增加未知混合流量的选项 -->
        <el-form-item label="流量类型(GT)">
          <el-select v-model="testForm.ground_truth" style="width: 100%;">
            <el-option label="外部未知混合流量 (仅筛查，不计准确率)" :value="-1" />
            <el-option label="纯正常流量包 (Benign)" :value="0" />
            <el-option label="纯恶意流量包 (Malware)" :value="1" />
          </el-select>
        </el-form-item>

        <el-form-item label="数据包 (PCAP)">
          <el-upload
            drag
            action="#"
            :auto-upload="false"
            :on-change="handleFileChange"
            :limit="1"
            accept=".pcap"
            class="cyber-upload"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text" style="color:var(--text-secondary)">
              拖拽 PCAP 文件至此，或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip" style="color:var(--accent-blue)">
                支持真实环境抓包，系统将自动切分流、提取特征并进行AI推理
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>

      <!-- 结果展示区 -->
      <div v-if="testResult" class="test-result-box">
         <h4 class="text-green" style="margin-top:0; border-bottom:1px solid #2d3748; padding-bottom:10px;">
           <el-icon><Monitor /></el-icon> 分析报告生成成功
         </h4>
         <div class="result-details">
           <p>解析文件: <span class="text-primary">{{ testResult.filename }}</span></p>
           <p>提取有效流: <span class="text-blue">{{ testResult.total_flows }} 个会话</span></p>
           <p>检出恶意流: <span class="text-red">{{ testResult.malicious_flows || 0 }} 个威胁</span></p>
           
           <!-- 有 Ground Truth 时显示准确率 -->
           <div v-if="testResult.accuracy !== null && testResult.accuracy !== undefined" class="acc-score">
             推理准确率 (Accuracy): 
             <span :class="testResult.accuracy > 80 ? 'text-green' : 'text-red'">
               {{ testResult.accuracy }}%
             </span>
           </div>
           
           <!-- 无 Ground Truth 时（混合包）显示筛查提示 -->
           <div v-else class="acc-score text-blue" style="font-size: 16px;">
             ✅ 混合流量筛查完毕，具体威胁已记录至后端日志。
           </div>
         </div>
      </div>

      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showOfflineTestDialog = false">关 闭</el-button>
          <el-button type="primary" @click="submitOfflineTest" :loading="isTesting">开始智能提取与分析</el-button>
        </span>
      </template>
    </el-dialog>

    <!-- ================= 弹窗：检测记录查询 ================= -->
    <el-dialog
      v-model="showRecordsDialog"
      title="检测记录查询"
      width="900px"
      custom-class="cyber-dialog"
      :close-on-click-modal="false"
    >
      <div style="margin-bottom: 12px;">
        <el-radio-group v-model="recordsFilter" size="small" @change="fetchRecords">
          <el-radio-button :label="null">全部</el-radio-button>
          <el-radio-button :label="1">实时嗅探</el-radio-button>
          <el-radio-button :label="0">离线分析</el-radio-button>
        </el-radio-group>
      </div>
      <el-table :data="records" style="width: 100%; background: transparent;" height="400" size="small">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="timestamp" label="时间" width="150" />
        <el-table-column prop="source" label="来源" width="100" />
        <el-table-column prop="flow_key" label="五元组" show-overflow-tooltip />
        <el-table-column prop="model_name" label="模型" width="140" />
        <el-table-column prop="prediction" label="判定" width="90">
          <template #default="scope">
            <span :class="scope.row.prediction === 'Malicious' ? 'tag-malicious' : 'tag-normal'">
              {{ scope.row.prediction }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="scope">
            {{ (scope.row.confidence * 100).toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column prop="packets_count" label="包数" width="70" />
      </el-table>
      <template #footer>
        <el-button @click="showRecordsDialog = false">关 闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { Platform, ArrowDown, TrendCharts, List, Monitor, UploadFilled } from '@element-plus/icons-vue'

// ================= 状态定义 =================
const stats = ref({ total_packets: 0, malicious_count: 0, start_time: '--:--' })
const logs = ref([])
const currentModelName = ref("Initializing...")
const chartRef = ref(null)
let myChart = null
let timer = null
const viewMode = ref('realtime')

// 模型性能统计
const modelStats = ref({ total_inferences: 0, malicious_count: 0, avg_confidence: 0 })

// 注意：确保后端的端口是 18000
const API_BASE = 'http://localhost:18000/api'

// === 离线测试状态 ===
const showOfflineTestDialog = ref(false)
const isTesting = ref(false)
const testForm = ref({ model_name: 'resnet', dataset_type: 'ustc', ground_truth: -1, file: null })
const testResult = ref(null)

// === 检测记录状态 ===
const showRecordsDialog = ref(false)
const records = ref([])
const recordsFilter = ref(null)

// ================= 业务逻辑 =================
const calcRate = (part, total) => {
  if (!total) return '0.00'
  return ((part / total) * 100).toFixed(2)
}

// === 切换在线模型 ===
const handleModelSwitch = async (command) => {
  try {
    ElMessage.info(`正在加载 ${command} 权重文件...`)
    const res = await axios.post(`${API_BASE}/switch_model`, { model_name: command })
    if (res.data.status === 'success') {
      ElMessage.success(`引擎已切换: ${res.data.current_model}`)
      fetchData()
    } else {
      ElMessage.error(res.data.message || '切换失败')
    }
  } catch (err) {
    ElMessage.error('无法连接到后端，请确保端口 18000 开放')
  }
}

// === 离线 PCAP 测试 ===
const handleFileChange = (uploadFile) => {
  testForm.value.file = uploadFile.raw
  testResult.value = null 
}

const submitOfflineTest = async () => {
  if (!testForm.value.file) {
    ElMessage.warning('请先选择一个 PCAP 文件！')
    return
  }

  isTesting.value = true
  testResult.value = null

  const formData = new FormData()
  formData.append('file', testForm.value.file)
  formData.append('model_name', testForm.value.model_name)
  formData.append('dataset_type', testForm.value.dataset_type)
  formData.append('ground_truth', testForm.value.ground_truth)

  try {
    const res = await axios.post(`${API_BASE}/upload_and_test`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    
    if (res.data.status === 'success') {
      ElMessage.success('流量解析与推理完成！')
      testResult.value = res.data
    } else {
      ElMessage.error(res.data.message || '分析过程中发生错误')
    }
  } catch (err) {
    ElMessage.error('请求超时或后端计算错误')
  } finally {
    isTesting.value = false
  }
}

// === 检测记录查询 ===
const openRecordsDialog = () => {
  showRecordsDialog.value = true
  fetchRecords()
}

const fetchRecords = async () => {
  try {
    const params = { limit: 100, offset: 0 }
    if (recordsFilter.value !== null) {
      params.realtime = recordsFilter.value
    }
    const res = await axios.get(`${API_BASE}/records`, { params })
    if (res.data.status === 'success') {
      records.value = res.data.records
    }
  } catch (err) {
    console.error('获取检测记录失败', err)
  }
}

// === 模型性能统计 ===
const fetchPerformance = async () => {
  try {
    const res = await axios.get(`${API_BASE}/performance`)
    if (res.data.status === 'success' && res.data.models.length > 0) {
      // 找当前模型的统计，若没有则取第一个
      const current = res.data.models.find(m => m.model_name === currentModelName.value)
        || res.data.models[0]
      modelStats.value = current
    }
  } catch (err) {
    // 静默失败
  }
}

// ================= ECharts & 轮询 =================
const initChart = () => {
  if (!chartRef.value) return
  myChart = echarts.init(chartRef.value, 'dark')
  myChart.setOption({
    backgroundColor: 'transparent', tooltip: { trigger: 'axis' },
    grid: { top: '15%', bottom: '10%', left: '10%', right: '5%' },
    xAxis: { type: 'category', boundaryGap: false, data:[], axisLine: { lineStyle: { color: '#555' } }, axisLabel: { color: '#94a3b8' } },
    yAxis: { type: 'value', name: 'Bytes', splitLine: { lineStyle: { color: '#333' } }, axisLabel: { color: '#94a3b8' } },
    series:[{
      name: 'Payload Size', type: 'line', smooth: true, symbol: 'none', itemStyle: { color: '#00f260' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1,[{ offset: 0, color: 'rgba(0, 242, 96, 0.3)' }, { offset: 1, color: 'rgba(0, 242, 96, 0.01)' }]) },
      data:[]
    }]
  })
}

const fetchData = async () => {
  if (isTesting.value) return

  try {
    const res = await axios.get(`${API_BASE}/dashboard`)
    const data = res.data
    
    if (data.stats) stats.value = data.stats
    if (data.current_model) currentModelName.value = data.current_model
    
    if (data.traffic_log && Array.isArray(data.traffic_log)) {
      logs.value = data.traffic_log.reverse()
      if (logs.value.length > 0 && myChart) {
        const recent = logs.value.slice(0, 50).reverse()
        myChart.setOption({
          xAxis: { data: recent.map(p => p.timestamp) },
          series:[{ data: recent.map(p => p.length) }]
        })
      }
    }
    
    // 顺带拉取模型性能统计
    fetchPerformance()
  } catch (err) {
    // 静默失败
  }
}

const tableRowClassName = ({ row }) => {
  return (row && row.prediction === 'Malicious') ? 'warning-row' : ''
}

onMounted(() => {
  initChart()
  fetchData()
  timer = setInterval(fetchData, 1000)
  window.addEventListener('resize', () => myChart && myChart.resize())
})

onUnmounted(() => {
  clearInterval(timer)
  if (myChart) myChart.dispose()
})
</script>

<style>
/* 保持你原本的样式不变 */
:root {
  --bg-color: #0b1120; --card-bg: #151e32; --text-primary: #e0e6ed; --text-secondary: #94a3b8;
  --accent-blue: #3b82f6; --accent-green: #00f260; --accent-red: #ef4444; --border-color: #2d3748;
}

body, html, #app {
  margin: 0; padding: 0; width: 100%; height: 100%;
  background-color: var(--bg-color); font-family: 'Inter', sans-serif; overflow: hidden;
}

.app-container { height: 100vh; display: flex; flex-direction: column; color: var(--text-primary); }

.navbar {
  height: 60px; background: rgba(21, 30, 50, 0.9); border-bottom: 1px solid var(--border-color);
  backdrop-filter: blur(10px); display: flex; justify-content: space-between; align-items: center; padding: 0 24px;
}
.brand { display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 700; color: var(--accent-blue); letter-spacing: 1px; }
.brand small { font-size: 12px; color: var(--text-secondary); opacity: 0.6; margin-left: 5px; }

.controls-area { display: flex; gap: 20px; align-items: center; }
.model-control { display: flex; align-items: center; gap: 10px; border-left: 1px solid var(--border-color); padding-left: 20px; }
.model-control .label { font-size: 14px; color: var(--text-secondary); }

.main-content { padding: 20px; flex: 1; display: flex; flex-direction: column; gap: 20px; }

.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
.cyber-card {
  background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; position: relative; overflow: hidden;
}
.cyber-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--accent-blue); }
.cyber-card:nth-child(2)::before { background: var(--accent-red); }
.cyber-card:nth-child(3)::before { background: var(--accent-green); }
.cyber-card:nth-child(4)::before { background: #a855f7; }

.card-title { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; }
.card-num { font-size: 32px; font-weight: 700; margin-top: 5px; font-family: 'Courier New', monospace; }
.card-sub { font-size: 12px; color: var(--text-secondary); margin-top: 5px; }
.text-blue { color: var(--accent-blue); }
.text-red { color: var(--accent-red); }
.text-green { color: var(--accent-green); }
.text-primary { color: var(--text-primary); }

.status-dot { width: 8px; height: 8px; background: var(--accent-green); border-radius: 50%; display: inline-block; box-shadow: 0 0 8px var(--accent-green); margin-right: 5px; }

.dashboard-grid { flex: 1; display: grid; grid-template-columns: 3fr 2fr; gap: 20px; min-height: 0; }
.cyber-panel { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
.panel-header { padding: 12px 20px; border-bottom: 1px solid var(--border-color); font-weight: 600; display: flex; align-items: center; gap: 8px; color: var(--text-secondary); background: rgba(0,0,0,0.1); }
.chart-container { flex: 1; width: 100%; height: 100%; }

.el-table { --el-table-bg-color: transparent; --el-table-tr-bg-color: transparent; --el-table-header-bg-color: rgba(0,0,0,0.2); --el-table-text-color: var(--text-secondary); --el-table-header-text-color: var(--text-primary); --el-table-border-color: #2d3748; }
.el-table td, .el-table th { border-bottom: 1px solid #2d3748 !important; }
.el-table--enable-row-hover .el-table__body tr:hover > td { background-color: rgba(255,255,255,0.05) !important; }
.tag-malicious { color: var(--accent-red); font-weight: bold; text-shadow: 0 0 5px rgba(239, 68, 68, 0.4); }
.tag-normal { color: var(--accent-green); }

.el-dialog.cyber-dialog { background: var(--card-bg) !important; border: 1px solid var(--border-color); border-radius: 8px; }
.cyber-dialog .el-dialog__title { color: var(--text-primary); font-weight: bold; }
.cyber-dialog .el-form-item__label { color: var(--text-secondary); }

.cyber-upload .el-upload-dragger { background-color: rgba(0,0,0,0.2); border-color: var(--border-color); }
.cyber-upload .el-upload-dragger:hover { border-color: var(--accent-blue); }

.test-result-box {
  margin-top: 20px; padding: 20px; border-radius: 8px;
  background: rgba(0, 0, 0, 0.3); border: 1px solid var(--accent-blue);
  box-shadow: inset 0 0 20px rgba(59, 130, 246, 0.1);
}
.result-details p { margin: 10px 0; color: var(--text-secondary); font-size: 14px; }
.acc-score { margin-top: 15px; font-size: 24px; font-weight: bold; color: var(--text-primary); border-top: 1px dashed var(--border-color); padding-top: 15px;}
</style>
