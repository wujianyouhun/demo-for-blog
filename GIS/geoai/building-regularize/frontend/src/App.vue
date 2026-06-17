<template>
  <div class="app-layout">
    <!-- ── 侧边栏 ── -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>建筑物轮廓正则化</h1>
        <p>GeoAI 提取结果 → GIS 制图规范</p>
      </div>

      <!-- 数据加载 -->
      <div class="sidebar-section">
        <h3>演示数据</h3>
        <div class="demo-btns">
          <button
            v-for="name in demoList" :key="name"
            class="btn btn-sm"
            :class="{ 'btn-primary': currentDemo === name }"
            @click="handleLoadDemo(name)"
          >
            {{ demoLabels[name] || name }}
          </button>
        </div>
        <div style="margin-top: 8px">
          <label class="btn btn-sm btn-block" style="cursor:pointer">
            上传 GeoJSON
            <input type="file" accept=".geojson,.json" @change="handleUpload"
              style="display:none" />
          </label>
        </div>
      </div>

      <!-- 参数配置 -->
      <div class="sidebar-section">
        <h3>参数配置</h3>

        <div class="param-group">
          <div class="param-label">
            <span>最小面积 (m²)</span>
            <span class="param-value">{{ config.min_area }}</span>
          </div>
          <input type="range" v-model.number="config.min_area" min="0" max="200" step="5" />
        </div>

        <div class="param-group">
          <div class="param-label">
            <span>简化容差 (m)</span>
            <span class="param-value">{{ config.dp_tolerance }}</span>
          </div>
          <input type="range" v-model.number="config.dp_tolerance" min="0.1" max="3" step="0.1" />
        </div>

        <div class="param-group">
          <div class="param-label">
            <span>角度吸附阈值 (°)</span>
            <span class="param-value">{{ config.angle_threshold }}</span>
          </div>
          <input type="range" v-model.number="config.angle_threshold" min="2" max="20" step="1" />
        </div>

        <div class="param-group">
          <div class="param-label">
            <span>平滑迭代</span>
            <span class="param-value">{{ config.smooth_iterations }}</span>
          </div>
          <input type="range" v-model.number="config.smooth_iterations" min="0" max="5" step="1" />
        </div>

        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="config.enable_symmetry" />
            <span>启用对称化</span>
          </label>
        </div>

        <button class="btn btn-primary btn-block" :disabled="!hasData || running"
          @click="handleRun">
          <span v-if="running" class="spinner" style="width:14px;height:14px;border-width:2px"></span>
          {{ running ? '运行中...' : '执行正则化' }}
        </button>
      </div>

      <!-- 步骤列表 -->
      <div class="sidebar-section" v-if="steps.length > 0">
        <h3>流水线步骤</h3>
        <div class="step-list">
          <div
            v-for="step in steps" :key="step.key"
            class="step-item"
            :class="{ active: activeStep === step.key }"
            @click="handleStepClick(step.key)"
          >
            <span class="step-badge">{{ stepIndex(step.key) }}</span>
            <span class="step-name">{{ step.label }}</span>
            <span class="step-count">{{ step.count }}</span>
          </div>
        </div>
      </div>

      <!-- 统计对比 -->
      <div class="sidebar-section" v-if="stats">
        <h3>正则化效果</h3>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{{ stats.before.count }}</div>
            <div class="stat-label">原始数量</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ stats.after.count }}</div>
            <div class="stat-label">过滤后数量</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ stats.before.avg_vertices }}</div>
            <div class="stat-label">原始平均顶点</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ stats.after.avg_vertices }}</div>
            <div class="stat-label">正则化平均顶点</div>
          </div>
        </div>
      </div>

      <!-- 导出 -->
      <div class="sidebar-section" v-if="steps.length > 0">
        <h3>导出结果</h3>
        <div style="display:flex;gap:6px">
          <button class="btn btn-sm btn-success" @click="handleExport('geojson')">GeoJSON</button>
          <button class="btn btn-sm" @click="handleExport('gpkg')">GeoPackage</button>
          <button class="btn btn-sm" @click="handleExport('shp')">Shapefile</button>
        </div>
      </div>
    </aside>

    <!-- ── 地图区 ── -->
    <div class="map-area">
      <div class="map-toolbar">
        <label class="layer-toggle">
          <input type="checkbox" v-model="showRaw" />
          <span style="color:#ef4444">■</span> 原始轮廓
        </label>
        <label class="layer-toggle">
          <input type="checkbox" v-model="showResult" />
          <span style="color:#2563eb">■</span> 当前步骤结果
        </label>
        <span v-if="currentDemo" style="margin-left:auto;color:var(--text-secondary);font-size:12px">
          {{ demoLabels[currentDemo] || currentDemo }}
        </span>
      </div>

      <MapView
        :rawGeojson="rawGeojson"
        :stepGeojson="activeStepGeojson"
        :showRaw="showRaw"
        :showResult="showResult"
      />

      <div class="status-bar">
        <span v-if="rawGeojson">原始: {{ rawGeojson.features?.length || 0 }} 个建筑</span>
        <span v-if="activeStepGeojson">当前步骤: {{ activeStepGeojson.features?.length || 0 }} 个</span>
        <span v-if="directions.length > 0">主方向: {{ directions.slice(0,5).map(d => d + '°').join(', ') }}{{ directions.length > 5 ? '...' : '' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import './assets/main.css'
import MapView from './components/MapView.vue'
import {
  listDemos, loadDemo, uploadGeoJSON,
  runPipeline, getCompareStats, exportResult,
} from './api/index.js'

// ── 状态 ──
const demoList = ref([])
const demoLabels = {
  residential: '住宅楼群',
  commercial: '商业建筑',
  mixed: '混合场景',
}
const currentDemo = ref('')
const hasData = ref(false)
const running = ref(false)

const rawGeojson = ref(null)
const steps = ref([])
const activeStep = ref('')
const activeStepGeojson = ref(null)
const directions = ref([])
const stats = ref(null)

const showRaw = ref(true)
const showResult = ref(true)

const config = reactive({
  min_area: 20,
  dp_tolerance: 0.5,
  use_pca: false,
  angle_threshold: 10,
  snap_angles: [0, 45, 90, 135],
  enable_symmetry: false,
  symmetry_tolerance: 2.0,
  smooth_iterations: 0,
  smooth_ratio: 0.25,
})

function stepIndex(key) {
  const idx = steps.value.findIndex(s => s.key === key)
  return idx >= 0 ? idx + 1 : '?'
}

// ── 初始化 ──
onMounted(async () => {
  try {
    const res = await listDemos()
    demoList.value = res.demos
  } catch (e) {
    console.error('无法获取演示数据列表', e)
  }
})

// ── 加载演示数据 ──
async function handleLoadDemo(name) {
  currentDemo.value = name
  try {
    const res = await loadDemo(name)
    rawGeojson.value = res.geojson
    hasData.value = true
    steps.value = []
    activeStep.value = ''
    activeStepGeojson.value = null
    stats.value = null
    directions.value = []
  } catch (e) {
    alert('加载失败: ' + (e.response?.data?.detail || e.message))
  }
}

// ── 上传 GeoJSON ──
async function handleUpload(e) {
  const file = e.target.files[0]
  if (!file) return
  currentDemo.value = ''
  try {
    const res = await uploadGeoJSON(file)
    rawGeojson.value = res.geojson
    hasData.value = true
    steps.value = []
    activeStep.value = ''
    activeStepGeojson.value = null
    stats.value = null
    directions.value = []
  } catch (e) {
    alert('上传失败: ' + (e.response?.data?.detail || e.message))
  }
  e.target.value = ''
}

// ── 执行流水线 ──
async function handleRun() {
  running.value = true
  try {
    const res = await runPipeline({ ...config })
    steps.value = res.steps
    directions.value = res.directions || []
    // 默认显示最终结果
    if (steps.value.length > 0) {
      const finalStep = steps.value[steps.value.length - 1]
      activeStep.value = finalStep.key
      activeStepGeojson.value = finalStep.geojson
    }
    // 获取统计
    try {
      stats.value = await getCompareStats()
    } catch (_) {}
  } catch (e) {
    alert('运行失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    running.value = false
  }
}

// ── 点击步骤 ──
function handleStepClick(key) {
  activeStep.value = key
  const step = steps.value.find(s => s.key === key)
  if (step) {
    activeStepGeojson.value = step.geojson
  }
}

// ── 导出 ──
async function handleExport(fmt) {
  try {
    await exportResult(fmt)
  } catch (e) {
    alert('导出失败: ' + (e.response?.data?.detail || e.message))
  }
}
</script>
