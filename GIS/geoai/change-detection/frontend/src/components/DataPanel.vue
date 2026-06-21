<template>
  <div class="panel-section">
    <h3 class="panel-title">数据下载</h3>
    <el-form :model="form" label-width="80px" size="default">
      <el-form-item label="下载区域">
        <el-select v-model="form.region" @change="onRegionChange">
          <el-option v-for="(v, k) in regions" :key="k" :label="v.name" :value="k" />
          <el-option label="自定义 BBox" value="custom" />
        </el-select>
      </el-form-item>

      <el-form-item label="BBox" v-if="form.region === 'custom'">
        <el-row :gutter="6">
          <el-col :span="6"><el-input-number v-model="form.bbox[0]" :precision="3" size="small" controls-position="right" /></el-col>
          <el-col :span="6"><el-input-number v-model="form.bbox[1]" :precision="3" size="small" controls-position="right" /></el-col>
          <el-col :span="6"><el-input-number v-model="form.bbox[2]" :precision="3" size="small" controls-position="right" /></el-col>
          <el-col :span="6"><el-input-number v-model="form.bbox[3]" :precision="3" size="small" controls-position="right" /></el-col>
        </el-row>
      </el-form-item>

      <el-form-item label="时相 A">
        <el-date-picker v-model="form.date_a" type="date" format="YYYY-MM-DD" value-format="YYYY-MM-DD" placeholder="选择日期" />
      </el-form-item>
      <el-form-item label="时相 B">
        <el-date-picker v-model="form.date_b" type="date" format="YYYY-MM-DD" value-format="YYYY-MM-DD" placeholder="选择日期" />
      </el-form-item>

      <el-form-item label="最大云量">
        <el-slider v-model="form.max_cloud_cover" :min="0" :max="100" show-input />
      </el-form-item>
    </el-form>

    <el-button type="primary" class="w-full" :loading="downloading" @click="download">
      <el-icon class="mr-1"><Download /></el-icon> 下载双时相影像
    </el-button>

    <!-- 下载进度 -->
    <div class="progress-section" v-if="downloadTask">
      <div class="progress-header">
        <span class="progress-stage">{{ stageLabel }}</span>
        <span class="progress-pct">{{ downloadTask.progress }}%</span>
      </div>
      <el-progress
        :percentage="downloadTask.progress"
        :status="progressStatus"
        :stroke-width="16"
        striped
        striped-flow
        :indeterminate="downloadTask.status === 'running' && downloadTask.progress === 0"
      />
      <div class="progress-message">{{ downloadTask.message }}</div>
    </div>

    <!-- 已有影像对 -->
    <div class="mt-4" v-if="pairs.length">
      <h4 class="text-sm font-semibold text-gray-600 mb-2">已有影像对 ({{ pairs.length }})</h4>
      <el-scrollbar max-height="220px">
        <div
          v-for="p in pairs" :key="p.name"
          class="pair-item"
          :class="{ active: currentPairName === p.name }"
          @click="selectPair(p)"
        >
          <div class="pair-name">{{ p.name }}</div>
          <div class="pair-size">A: {{ fmtSize(p.size_a) }} / B: {{ fmtSize(p.size_b) }}</div>
        </div>
      </el-scrollbar>
    </div>

    <el-button size="small" text class="mt-2" @click="refreshPairs">
      <el-icon><Refresh /></el-icon> 刷新
    </el-button>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api/client.js'

const emit = defineEmits(['task', 'pair-loaded', 'download-progress'])
const downloading = ref(false)
const pairs = ref([])
const regions = ref({})
const downloadTask = ref(null)
const currentPairName = ref(null)
let pollTimer = null

const form = reactive({
  region: 'beijing',
  bbox: [116.2, 39.75, 116.6, 40.05],
  date_a: '2022-06-01',
  date_b: '2023-06-01',
  max_cloud_cover: 20,
})

const regionBbox = {
  beijing: [116.2, 39.75, 116.6, 40.05],
  shanghai: [121.4, 31.1, 121.7, 31.35],
  shenzhen: [113.85, 22.45, 114.15, 22.65],
  dubai: [55.1, 25.05, 55.4, 25.25],
  san_francisco: [-122.52, 37.7, -122.35, 37.83],
}

const stageLabels = {
  init: '准备中...',
  searching: '搜索可用影像...',
  downloading_a: '下载时相 A...',
  downloading_b: '下载时相 B...',
  aligning: '空间对齐中...',
  done: '下载完成',
  error: '下载失败',
}

const stageLabel = computed(() => {
  if (!downloadTask.value) return ''
  return stageLabels[downloadTask.value.stage] || downloadTask.value.message || ''
})

const progressStatus = computed(() => {
  if (!downloadTask.value) return ''
  if (downloadTask.value.status === 'completed') return 'success'
  if (downloadTask.value.status === 'failed') return 'exception'
  return ''
})

onMounted(async () => {
  try { const r = await api.get('/api/data/regions'); regions.value = r.data } catch {}
  refreshPairs()
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

function onRegionChange(r) {
  if (r !== 'custom' && regionBbox[r]) form.bbox = [...regionBbox[r]]
}

async function download() {
  downloading.value = true
  downloadTask.value = { status: 'running', progress: 0, stage: 'init', message: '正在提交下载任务...' }
  try {
    const r = await api.post('/api/data/download-pair', form)
    ElMessage.success('下载任务已启动')
    emit('task', { status: 'running', message: '正在下载双时相影像...' })
    startPolling(r.data.task_id)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || e.message)
    downloadTask.value = { status: 'failed', progress: 0, stage: 'error', message: e.message }
  } finally {
    downloading.value = false
  }
}

function startPolling(taskId) {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      const r = await api.get(`/api/data/download-pair/${taskId}`)
      const task = r.data
      downloadTask.value = task
      emit('download-progress', task)

      if (task.status === 'completed') {
        clearInterval(pollTimer)
        pollTimer = null
        ElMessage.success(task.message)
        emit('task', { status: 'success', message: task.message })
        if (task.result) {
          emit('pair-loaded', task.result)
        }
        refreshPairs()
      } else if (task.status === 'failed') {
        clearInterval(pollTimer)
        pollTimer = null
        ElMessage.error(task.message)
        emit('task', { status: 'error', message: task.message })
      }
    } catch {}
  }, 2000)
}

function selectPair(p) {
  currentPairName.value = p.name
  emit('pair-loaded', p)
}

async function refreshPairs() {
  try { const r = await api.get('/api/data/pairs'); pairs.value = r.data.pairs || [] } catch { pairs.value = [] }
}

function fmtSize(b) {
  if (!b) return '0 B'
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0
  while (b >= 1024 && i < 3) { b /= 1024; i++ }
  return b.toFixed(1) + ' ' + u[i]
}
</script>

<style scoped>
.pair-item {
  padding: 8px 12px; border-bottom: 1px solid #f5f5f5; cursor: pointer;
  transition: background 0.15s;
}
.pair-item:hover { background: #f0f7ff; }
.pair-item.active { background: #e6f0ff; border-left: 3px solid #409eff; }
.pair-name { font-size: 13px; font-weight: 500; color: #303133; }
.pair-size { font-size: 11px; color: #909399; margin-top: 2px; }

.progress-section {
  margin-top: 16px; padding: 12px; background: #f9fafb; border-radius: 8px;
}
.progress-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 6px;
}
.progress-stage { font-size: 13px; font-weight: 500; color: #303133; }
.progress-pct { font-size: 13px; color: #909399; }
.progress-message { font-size: 12px; color: #909399; margin-top: 4px; }
</style>
