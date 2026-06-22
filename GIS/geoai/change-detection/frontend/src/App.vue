<template>
  <el-container style="height: 100vh;">
    <el-header class="header-bar">
      <div class="header-left">
        <el-icon :size="26" color="#fff"><Odometer /></el-icon>
        <h1 class="app-title">ChangeDetection</h1>
        <span class="app-subtitle">遥感影像时序变化检测平台</span>
      </div>
      <el-tag :type="apiOk ? 'success' : 'danger'" size="small">
        {{ apiOk ? 'API 已连接' : 'API 未连接' }}
      </el-tag>
    </el-header>

    <el-container>
      <el-aside width="400px" class="side-panel border-r">
        <el-tabs v-model="tab" type="border-card" class="h-full">
          <el-tab-pane label="数据下载" name="data">
            <DataPanel
              @task="onTask"
              @pair-loaded="onPairLoaded"
              @download-progress="onDownloadProgress"
            />
          </el-tab-pane>
          <el-tab-pane label="变化检测" name="detect">
            <DetectionPanel
              :current-pair="currentPair"
              @task="onTask"
              @result="onDetectResult"
            />
          </el-tab-pane>
          <el-tab-pane label="对比分析" name="compare">
            <ComparePanel
              :current-pair="currentPair"
              :detect-result="detectResult"
              @task="onTask"
              @show-geojson="onShowGeoJson"
            />
          </el-tab-pane>
        </el-tabs>
      </el-aside>

      <el-main class="p-0 relative">
        <MapCompare
          ref="mapRef"
          :compare-mode="compareMode"
          :geojson-data="geojsonData"
          :image-url-a="imageUrlA"
          :image-url-b="imageUrlB"
          :image-bounds="imageBounds"
          :change-map-url="changeMapUrl"
        />

        <!-- 模式切换按钮 -->
        <div class="mode-bar">
          <el-radio-group v-model="compareMode" size="small">
            <el-radio-button value="swipe">卷帘</el-radio-button>
            <el-radio-button value="side">并列</el-radio-button>
            <el-radio-button value="change">变化图</el-radio-button>
          </el-radio-group>
        </div>

        <!-- 通知 -->
        <div class="task-toast" v-if="tasks.length">
          <el-card v-for="t in tasks" :key="t.id" shadow="hover" class="mb-2 text-sm">
            <span :class="'status-' + t.status">{{ t.message }}</span>
          </el-card>
        </div>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, reactive, onMounted, watch } from 'vue'
import api from './api/client.js'
import MapCompare from './components/MapCompare.vue'
import DataPanel from './components/DataPanel.vue'
import DetectionPanel from './components/DetectionPanel.vue'
import ComparePanel from './components/ComparePanel.vue'

const tab = ref('data')
const apiOk = ref(false)
const compareMode = ref('swipe')
const currentPair = ref(null)
const detectResult = ref(null)
const geojsonData = ref(null)
const mapRef = ref(null)
const tasks = reactive([])
let tid = 0

// 影像预览 URL
const imageUrlA = ref(null)
const imageUrlB = ref(null)
const imageBounds = ref(null)
const changeMapUrl = ref(null)

onMounted(async () => {
  try {
    const r = await api.get('/api/health')
    apiOk.value = r.data.status === 'ok'
  } catch { apiOk.value = false }
})

function onTask(t) {
  const id = ++tid
  tasks.unshift({ id, ...t })
  if (tasks.length > 4) tasks.pop()
  setTimeout(() => { const i = tasks.findIndex(x => x.id === id); if (i >= 0) tasks.splice(i, 1) }, 30000)
}

function onDownloadProgress(task) {
  // 可扩展：根据下载进度在地图上做提示
}

/**
 * 构建 GeoTIFF 预览 URL。
 * 支持 data/raw/time_a、data/raw/time_b 和 data/output 下的结果图。
 */
function buildPreviewUrl(filePath) {
  if (!filePath) return null
  return `/api/data/preview-file?path=${encodeURIComponent(filePath)}`
}

async function fetchBounds(previewUrl) {
  if (!previewUrl) return null
  try {
    const r = await api.head(previewUrl)
    const boundsStr = r.headers['x-image-bounds']
    if (boundsStr) {
      return boundsStr.split(',').map(Number)
    }
  } catch {}
  return null
}

async function onPairLoaded(pair) {
  currentPair.value = pair
  tab.value = 'detect'

  // 构建预览 URL 并传给地图
  const urlA = buildPreviewUrl(pair.time_a)
  const urlB = buildPreviewUrl(pair.time_b)
  imageUrlA.value = urlA
  imageUrlB.value = urlB

  // 获取影像范围 (bounds)
  if (urlA) {
    const bounds = await fetchBounds(urlA)
    if (bounds) {
      imageBounds.value = bounds
    }
  }
}

function onDetectResult(r) {
  detectResult.value = r
  tab.value = 'compare'
  if (r.mask) {
    changeMapUrl.value = buildPreviewUrl(r.mask)
  }
}

function onShowGeoJson(gj) {
  geojsonData.value = gj
}
</script>

<style scoped>
.header-bar {
  background: linear-gradient(135deg, #1a73e8, #0d47a1);
  color: #fff; display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px; height: 52px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.header-left { display: flex; align-items: center; gap: 10px; }
.app-title { font-size: 18px; font-weight: 700; margin: 0; }
.app-subtitle { font-size: 12px; color: rgba(255,255,255,0.7); }
.mode-bar { position: absolute; top: 12px; left: 12px; z-index: 1000; }
.task-toast { position: absolute; top: 50px; right: 12px; z-index: 1000; max-width: 340px; }
</style>
