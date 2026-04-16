<template>
    <div class="cyber-panel chart-panel">
      <div class="panel-header">
        <el-icon><TrendCharts /></el-icon> 实时流量载荷分布
        <el-tag size="small" type="info" effect="dark" style="margin-left: auto;">
          更新中 {{ new Date().toLocaleTimeString() }}
        </el-tag>
      </div>
      <div ref="chartRef" class="chart-container"></div>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted, onUnmounted, watch } from 'vue'
  import * as echarts from 'echarts'
  import { TrendCharts } from '@element-plus/icons-vue'
  
  const props = defineProps({
    logs: { type: Array, default: () => [] }
  })
  
  const chartRef = ref(null)
  let chartInstance = null
  
  const initChart = () => {
    if (!chartRef.value) return
    chartInstance = echarts.init(chartRef.value, 'dark')
    chartInstance.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(21,30,50,0.9)',
        borderColor: '#00e5b0',
        textStyle: { color: '#e0e6ed' }
      },
      grid: { top: 30, bottom: 30, left: 50, right: 20 },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: [],
        axisLine: { lineStyle: { color: '#555' } },
        axisLabel: { color: '#94a3b8' }
      },
      yAxis: {
        type: 'value',
        name: 'Bytes',
        nameTextStyle: { color: '#94a3b8' },
        splitLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#94a3b8' }
      },
      series: [{
        name: 'Payload Size',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { width: 2, color: '#00e5b0' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(0, 229, 176, 0.6)' },
            { offset: 0.8, color: 'rgba(0, 229, 176, 0.02)' }
          ])
        },
        data: []
      }]
    })
    window.addEventListener('resize', () => chartInstance?.resize())
  }
  
  const updateChart = (logsArray) => {
    if (!chartInstance) return
    if (logsArray.length === 0) {
      chartInstance.setOption({
        title: { show: true, text: '暂无流量数据', left: 'center', top: 'center', textStyle: { color: '#94a3b8' } },
        series: [{ data: [] }],
        xAxis: { data: [] }
      })
    } else {
      const recent = logsArray.slice(0, 50).reverse()
      chartInstance.setOption({
        title: { show: false },
        xAxis: { data: recent.map(p => p.timestamp) },
        series: [{ data: recent.map(p => p.length) }]
      })
    }
  }
  
  watch(() => props.logs, (newLogs) => {
    updateChart(newLogs)
  }, { deep: false, immediate: true })
  
  onMounted(() => {
    initChart()
  })
  
  onUnmounted(() => {
    chartInstance?.dispose()
  })
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
  
  .chart-container {
    flex: 1;
    width: 100%;
    min-height: 200px;
  }
  </style>