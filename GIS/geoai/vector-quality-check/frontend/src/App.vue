<template>
  <div class="app-layout">
    <!-- ── 侧边栏 ── -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>矢量数据质量自动检查</h1>
        <p>GeoAI 拓扑错误检测 → 一键修复</p>
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

      <!-- 质量检查 -->
      <div class="sidebar-section">
        <h3>质量检查参数</h3>

        <div class="param-group">
          <div class="param-label">
            <span>碎片最小面积 (m²)</span>
            <span class="param-value">{{ checkParams.sliver_min_area }}</span>
          </div>
          <input type="range" v-model.number="checkParams.sliver_min_area" min="1" max="50" step="1" />
        </div>

        <div class="param-group">
          <div class="param-label">
            <span>重叠检测阈值</span>
            <span class="param-value">{{ checkParams.overlap_threshold }}</span>
          </div>
          <input type="range" v-model.number="checkParams.overlap_threshold" min="0.01" max="0.5" step="0.01" />
        </div>

        <button class="btn btn-primary btn-block" :disabled="!hasData || checking"
          @click="handleCheck">
          <span v-if="checking" class="spinner" style="width:14px;height:14px;border-width:2px"></span>
          {{ checking ? '检查中...' : '执行质量检查' }}
        </button>
      </div>

      <!-- 检查报告 -->
      <div class="sidebar-section" v-if="report && report.total_issues !== undefined">
        <h3>检查报告</h3>

        <div class="report-summary">
          <div class="report-stat total">
            <div class="value">{{ report.total_issues }}</div>
            <div class="label">总问题数</div>
          </div>
          <div class="report-stat high">
            <div class="value">{{ report.by_severity?.HIGH || 0 }}</div>
            <div class="label">严重</div>
          </div>
          <div class="report-stat medium">
            <div class="value">{{ (report.by_severity?.MEDIUM || 0) + (report.by_severity?.LOW || 0) }}</div>
            <div class="label">中/低级</div>
          </div>
        </div>

        <!-- 类型分布 -->
        <div class="type-bars" v-if="report.by_type">
          <div class="type-bar" v-for="(count, type) in report.by_type" :key="type">
            <span class="bar-label">{{ typeLabels[type] || type }}</span>
            <div class="bar-track">
              <div class="bar-fill" :style="{ width: barWidth(count) + '%', background: typeColor(type) }"></div>
            </div>
            <span class="bar-count">{{ count }}</span>
          </div>
        </div>

        <!-- 问题列表 -->
        <div class="issue-list" v-if="issues.length > 0">
          <div
            v-for="(issue, i) in issues" :key="i"
            class="issue-item"
            :class="{ active: activeIssueIndex === i }"
            @click="handleIssueClick(issue, i)"
          >
            <span class="severity-dot" :class="issue.severity"></span>
            <span class="issue-type">{{ issue.error_type }}</span>
            <span class="issue-detail">{{ issue.detail }}</span>
          </div>
        </div>
      </div>

      <!-- 一键修复 -->
      <div class="sidebar-section" v-if="report && report.total_issues > 0">
        <h3>修复配置</h3>

        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="repairConfig.repair_invalid" />
            <span>修复无效几何</span>
          </label>
        </div>
        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="repairConfig.fill_holes" />
            <span>填充孔洞</span>
          </label>
        </div>
        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="repairConfig.remove_overlaps" />
            <span>去除重叠</span>
          </label>
        </div>
        <div class="param-group">
          <div class="param-label">
            <span>重叠 IoU 阈值</span>
            <span class="param-value">{{ repairConfig.overlap_threshold }}</span>
          </div>
          <input type="range" v-model.number="repairConfig.overlap_threshold" min="0.1" max="0.9" step="0.05" />
        </div>
        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="repairConfig.remove_slivers" />
            <span>去除碎片</span>
          </label>
        </div>
        <div class="param-group">
          <div class="param-label">
            <span>碎片面积阈值 (m²)</span>
            <span class="param-value">{{ repairConfig.min_area }}</span>
          </div>
          <input type="range" v-model.number="repairConfig.min_area" min="1" max="100" step="1" />
        </div>
        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="repairConfig.explode_multipart" />
            <span>拆分多部件</span>
          </label>
        </div>
        <div class="param-group">
          <label class="layer-toggle">
            <input type="checkbox" v-model="repairConfig.remove_duplicates" />
            <span>去除重复</span>
          </label>
        </div>

        <button class="btn btn-success btn-block" :disabled="repairing"
          @click="handleRepair" style="margin-top: 8px">
          <span v-if="repairing" class="spinner" style="width:14px;height:14px;border-width:2px;border-top-color:#fff"></span>
          {{ repairing ? '修复中...' : '一键修复' }}
        </button>
      </div>

      <!-- 修复步骤 -->
      <div class="sidebar-section" v-if="repairSteps.length > 0">
        <h3>修复步骤</h3>
        <div class="step-list">
          <div
            v-for="step in repairSteps" :key="step.key"
            class="step-item"
            :class="{ active: activeStep === step.key }"
            @click="handleStepClick(step.key)"
          >
            <span class="step-badge">{{ stepIndex(step.key) }}</span>
            <span class="step-name">{{ step.label }}</span>
            <span class="step-count">{{ stepCount(step) }}</span>
          </div>
        </div>
      </div>

      <!-- 修复日志 -->
      <div class="sidebar-section" v-if="repairLog.length > 0">
        <h3>修复日志</h3>
        <div class="repair-log">
          <div v-for="(line, i) in repairLog" :key="i">
            <span class="log-step">{{ line }}</span>
          </div>
        </div>
      </div>

      <!-- 统计对比 -->
      <div class="sidebar-section" v-if="repairResult">
        <h3>修复效果</h3>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{{ repairResult.before_count }}</div>
            <div class="stat-label">修复前数量</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ repairResult.after_count }}</div>
            <div class="stat-label">修复后数量</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ report.total_issues }}</div>
            <div class="stat-label">检出问题</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ repairSteps.length }}</div>
            <div class="stat-label">修复步骤</div>
          </div>
        </div>
      </div>

      <!-- 导出 -->
      <div class="sidebar-section" v-if="repairResult">
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
          <span style="color:#64748b">■</span> 原始数据
        </label>
        <label class="layer-toggle" v-if="issuesGeojson">
          <input type="checkbox" v-model="showIssues" />
          <span style="color:#ef4444">■</span> 问题标记
        </label>
        <label class="layer-toggle" v-if="repairedGeojson">
          <input type="checkbox" v-model="showRepair" />
          <span style="color:#10b981">■</span> 修复结果
        </label>
        <span v-if="currentDemo" style="margin-left:auto;color:var(--text-secondary);font-size:12px">
          {{ demoLabels[currentDemo] || currentDemo }}
        </span>
      </div>

      <MapView
        :rawGeojson="rawGeojson"
        :issuesGeojson="issuesGeojson"
        :repairGeojson="activeStepGeojson || repairedGeojson"
        :showRaw="showRaw"
        :showIssues="showIssues"
        :showRepair="showRepair"
        :highlightIssue="highlightIssue"
      />

      <div class="status-bar">
        <span v-if="rawGeojson">原始: {{ rawGeojson.features?.length || 0 }} 个要素</span>
        <span v-if="issues.length > 0">问题: {{ issues.length }} 个</span>
        <span v-if="repairedGeojson">修复后: {{ repairedGeojson.features?.length || 0 }} 个要素</span>
        <span v-if="activeStep">当前步骤: {{ activeStepLabel }}</span>
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
  runCheck, runRepair, exportResult,
} from './api/index.js'

// ── 状态 ──
const demoList = ref([])
const demoLabels = {
  ai_segmentation: 'AI分割结果',
  building_footprints: '建筑轮廓',
  mixed_errors: '混合错误',
}
const typeLabels = {
  'Invalid Geometry': '无效几何',
  'Self-Intersection': '自相交',
  'Hole': '孔洞',
  'Overlap': '重叠',
  'Sliver': '碎片',
  'MultiPart': '多部件',
  'Duplicate': '重复',
}

const currentDemo = ref('')
const hasData = ref(false)
const checking = ref(false)
const repairing = ref(false)

const rawGeojson = ref(null)
const issues = ref([])
const report = ref(null)
const issuesGeojson = ref(null)
const repairedGeojson = ref(null)
const repairSteps = ref([])
const repairLog = ref([])
const repairResult = ref(null)
const activeStep = ref('')
const activeStepGeojson = ref(null)
const activeIssueIndex = ref(-1)
const highlightIssue = ref(null)

const showRaw = ref(true)
const showIssues = ref(true)
const showRepair = ref(true)

const checkParams = reactive({
  sliver_min_area: 5.0,
  overlap_threshold: 0.05,
})

const repairConfig = reactive({
  repair_invalid: true,
  fill_holes: true,
  max_hole_area: null,
  remove_overlaps: true,
  overlap_method: 'area',
  overlap_threshold: 0.4,
  remove_slivers: true,
  min_area: 10,
  explode_multipart: true,
  remove_duplicates: true,
  simplify: false,
  simplify_tolerance: 0.5,
  preserve_topology: true,
})

const activeStepLabel = computed(() => {
  const step = repairSteps.value.find(s => s.key === activeStep.value)
  return step ? step.label : ''
})

function stepIndex(key) {
  const idx = repairSteps.value.findIndex(s => s.key === key)
  return idx >= 0 ? idx + 1 : '?'
}

function stepCount(step) {
  if (step.geojson && step.geojson.features) {
    return step.geojson.features.length + ' 个'
  }
  return ''
}

function barWidth(count) {
  if (!report.value || !report.value.total_issues) return 0
  return (count / report.value.total_issues) * 100
}

function typeColor(type) {
  const colors = {
    'Invalid Geometry': '#ef4444',
    'Self-Intersection': '#f43f5e',
    'Hole': '#f59e0b',
    'Overlap': '#e11d48',
    'Sliver': '#6b7280',
    'MultiPart': '#8b5cf6',
    'Duplicate': '#0ea5e9',
  }
  return colors[type] || '#64748b'
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
    _resetCheckState()
    _resetRepairState()
  } catch (e) {
    alert('加载失败: ' + (e.response?.data?.detail || e.message))
  }
}

// ── 上传 ──
async function handleUpload(e) {
  const file = e.target.files[0]
  if (!file) return
  currentDemo.value = ''
  try {
    const res = await uploadGeoJSON(file)
    rawGeojson.value = res.geojson
    hasData.value = true
    _resetCheckState()
    _resetRepairState()
  } catch (e) {
    alert('上传失败: ' + (e.response?.data?.detail || e.message))
  }
  e.target.value = ''
}

// ── 执行检查 ──
async function handleCheck() {
  checking.value = true
  _resetRepairState()
  try {
    const res = await runCheck(checkParams)
    issues.value = res.issues
    report.value = res.report
    issuesGeojson.value = res.issues_geojson
    activeIssueIndex.value = -1
    highlightIssue.value = null
  } catch (e) {
    alert('检查失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    checking.value = false
  }
}

// ── 点击问题 ──
function handleIssueClick(issue, index) {
  activeIssueIndex.value = index
  // 构造高亮几何
  if (issuesGeojson.value && issuesGeojson.value.features) {
    const match = issuesGeojson.value.features.find(f =>
      f.properties?.fid === String(issue.fid) && f.properties?.error_type === issue.error_type
    )
    if (match) {
      highlightIssue.value = { ...issue, geometry: match.geometry }
    }
  }
}

// ── 执行修复 ──
async function handleRepair() {
  repairing.value = true
  try {
    const res = await runRepair(repairConfig)
    repairSteps.value = res.steps
    repairLog.value = res.repair_log
    repairedGeojson.value = res.repaired_geojson
    repairResult.value = {
      before_count: res.before_count,
      after_count: res.after_count,
    }
    // 默认显示最终结果
    if (repairSteps.value.length > 0) {
      const finalStep = repairSteps.value[repairSteps.value.length - 1]
      activeStep.value = finalStep.key
      activeStepGeojson.value = null  // 使用 repairedGeojson
    }
    // 关闭问题标记, 显示修复结果
    showIssues.value = false
    showRepair.value = true
  } catch (e) {
    alert('修复失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    repairing.value = false
  }
}

// ── 点击步骤 ──
function handleStepClick(key) {
  activeStep.value = key
  const step = repairSteps.value.find(s => s.key === key)
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

function _resetCheckState() {
  issues.value = []
  report.value = null
  issuesGeojson.value = null
  activeIssueIndex.value = -1
  highlightIssue.value = null
}

function _resetRepairState() {
  repairedGeojson.value = null
  repairSteps.value = []
  repairLog.value = []
  repairResult.value = null
  activeStep.value = ''
  activeStepGeojson.value = null
}
</script>
