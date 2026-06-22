<template>
  <div class="app-layout">
    <!-- ── 侧边栏 ── -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>GeoAI 要素提取</h1>
        <p>遥感影像 → 建筑/林地/草地 矢量要素</p>
      </div>

      <!-- TIF 信息 -->
      <div class="sidebar-section" v-if="tifInfo">
        <h3>影像信息</h3>
        <div class="info-grid">
          <div class="info-item"><span class="label">文件</span><br><span class="value">{{ tifInfo.file_name }}</span></div>
          <div class="info-item"><span class="label">大小</span><br><span class="value">{{ tifInfo.file_size_mb }} MB</span></div>
          <div class="info-item"><span class="label">分辨率</span><br><span class="value">~{{ tifInfo.resolution?.approx_meters }}m</span></div>
          <div class="info-item"><span class="label">像素</span><br><span class="value">{{ tifInfo.pixel_count_millions }}M</span></div>
        </div>
        <button class="btn btn-sm btn-block" style="margin-top:8px" @click="handleGenTiles" :disabled="generatingTiles">
          {{ generatingTiles ? '生成中...' : '预生成瓦片' }}
        </button>
      </div>

      <!-- ROI 选择 -->
      <div class="sidebar-section">
        <h3>提取范围</h3>
        <div style="display:flex;gap:6px;margin-bottom:8px">
          <button class="btn btn-sm" :class="{ 'btn-primary': drawMode }" @click="toggleDraw">
            {{ drawMode ? '取消框选' : '框选 ROI' }}
          </button>
          <button class="btn btn-sm" v-if="roi" @click="clearROI">清除 ROI</button>
          <button class="btn btn-sm" :class="{ active: useFullImage }"
            @click="useFullImage = !useFullImage; roi = null">
            全图提取
          </button>
        </div>
        <div class="roi-info" v-if="roi">
          ROI: {{ roi.left.toFixed(4) }}°E ~ {{ roi.right.toFixed(4) }}°E,
          {{ roi.bottom.toFixed(4) }}°N ~ {{ roi.top.toFixed(4) }}°N
        </div>
        <div class="roi-info" v-else-if="useFullImage">
          使用完整影像范围进行提取
        </div>
        <div class="roi-info" v-else style="background:#fef3c7;border-color:#fcd34d;color:#92400e">
          请在地图上框选提取区域，或选择"全图提取"
        </div>
      </div>

      <!-- 提取目标 -->
      <div class="sidebar-section">
        <h3>提取目标</h3>
        <div class="target-chips">
          <div class="target-chip building" :class="{ active: targets.includes('building') }"
            @click="toggleTarget('building')">建筑</div>
          <div class="target-chip forest" :class="{ active: targets.includes('forest') }"
            @click="toggleTarget('forest')">林地</div>
          <div class="target-chip grassland" :class="{ active: targets.includes('grassland') }"
            @click="toggleTarget('grassland')">草地</div>
        </div>

        <h3 style="margin-top:4px">提取方法</h3>
        <div class="method-btns">
          <div class="method-btn geoai-btn" :class="{ active: method === 'geoai' }" @click="method = 'geoai'">
            GeoAI <span class="recommend-tag">推荐</span>
          </div>
          <div class="method-btn" :class="{ active: method === 'cv' }" @click="method = 'cv'">
            传统 CV
          </div>
          <div class="method-btn" :class="{ active: method === 'dl' }" @click="method = 'dl'">
            深度学习
          </div>
          <div class="method-btn" :class="{ active: method === 'hybrid' }" @click="method = 'hybrid'">
            混合
          </div>
        </div>
        <div class="method-desc" v-if="method === 'geoai'">
          使用 <b>geoai-py</b> 库: BuildingFootprintExtractor（建筑）+ GroundedSAM（林地/草地零样本分割）
        </div>
        <div class="method-desc" v-else-if="method === 'cv'">
          传统计算机视觉: Canny边缘检测 + HSV颜色阈值 + 形态学操作
        </div>
      </div>

      <!-- 参数配置 -->
      <div class="sidebar-section">
        <h3>参数配置</h3>

        <!-- GeoAI 参数 -->
        <template v-if="method === 'geoai'">
          <div class="param-group">
            <div class="param-label">
              <span>推理设备</span>
              <span class="param-value">{{ params.geoai_device }}</span>
            </div>
            <select v-model="params.geoai_device" class="param-select">
              <option value="cpu">CPU</option>
              <option value="cuda:0">CUDA:0</option>
              <option value="cuda:1">CUDA:1</option>
            </select>
          </div>
          <div class="param-group">
            <div class="param-label">
              <span>建筑正则化</span>
              <span class="param-value">{{ params.geoai_building_regularize ? '开启' : '关闭' }}</span>
            </div>
            <input type="checkbox" v-model="params.geoai_building_regularize" />
          </div>
          <div class="param-group">
            <div class="param-label">
              <span>GroundedSAM 置信度</span>
              <span class="param-value">{{ params.geoai_groundedsam_threshold }}</span>
            </div>
            <input type="range" v-model.number="params.geoai_groundedsam_threshold" min="0.1" max="0.9" step="0.05" />
          </div>
          <div class="param-group">
            <div class="param-label">
              <span>建筑 Batch Size</span>
              <span class="param-value">{{ params.geoai_building_batch_size }}</span>
            </div>
            <input type="range" v-model.number="params.geoai_building_batch_size" min="1" max="8" step="1" />
          </div>
        </template>

        <!-- CV 参数 -->
        <template v-else-if="targets.includes('building')">
          <div class="param-group">
            <div class="param-label">
              <span>Canny 低阈值</span>
              <span class="param-value">{{ params.building_canny_low }}</span>
            </div>
            <input type="range" v-model.number="params.building_canny_low" min="10" max="200" step="5" />
          </div>
          <div class="param-group">
            <div class="param-label">
              <span>Canny 高阈值</span>
              <span class="param-value">{{ params.building_canny_high }}</span>
            </div>
            <input type="range" v-model.number="params.building_canny_high" min="50" max="300" step="5" />
          </div>
          <div class="param-group">
            <div class="param-label">
              <span>建筑最小面积 (px)</span>
              <span class="param-value">{{ params.building_min_area_px }}</span>
            </div>
            <input type="range" v-model.number="params.building_min_area_px" min="20" max="1000" step="20" />
          </div>
        </template>

        <template v-if="method !== 'geoai' && targets.includes('forest')">
          <div class="param-group">
            <div class="param-label">
              <span>林地 HSV 色调下限</span>
              <span class="param-value">{{ params.forest_hsv_h_range[0] }}</span>
            </div>
            <input type="range" v-model.number="params.forest_hsv_h_range[0]" min="0" max="90" step="5" />
          </div>
          <div class="param-group">
            <div class="param-label">
              <span>林地 HSV 色调上限</span>
              <span class="param-value">{{ params.forest_hsv_h_range[1] }}</span>
            </div>
            <input type="range" v-model.number="params.forest_hsv_h_range[1]" min="50" max="180" step="5" />
          </div>
        </template>

        <div class="param-group">
          <div class="param-label">
            <span>简化容差</span>
            <span class="param-value">{{ params.simplify_tolerance }}</span>
          </div>
          <input type="range" v-model.number="params.simplify_tolerance" min="0" max="5" step="0.5" />
        </div>

        <div class="param-group">
          <div class="param-label">
            <span>最小多边形面积 (m²)</span>
            <span class="param-value">{{ params.min_polygon_area }}</span>
          </div>
          <input type="range" v-model.number="params.min_polygon_area" min="1" max="100" step="1" />
        </div>
      </div>

      <!-- 执行提取 -->
      <div class="sidebar-section">
        <button class="btn btn-primary btn-block" :disabled="extracting || targets.length === 0"
          @click="handleExtract" style="padding:12px">
          <span v-if="extracting" class="spinner" style="width:14px;height:14px;border-width:2px;border-top-color:#fff"></span>
          {{ extracting ? '提取中...' : `提取 ${targets.join(' + ')}` }}
        </button>
      </div>

      <!-- 提取结果 -->
      <div class="sidebar-section" v-if="stats">
        <h3>提取结果</h3>

        <div class="stats-grid" style="margin-bottom:12px">
          <div class="stat-card">
            <div class="stat-value">{{ totalFeatures }}</div>
            <div class="stat-label">提取要素数</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ stats.elapsed_seconds || 0 }}s</div>
            <div class="stat-label">耗时</div>
          </div>
        </div>

        <div class="result-summary">
          <div class="result-class" v-for="target in Object.keys(stats.targets || {})" :key="target">
            <span class="dot" :style="{ background: classColorMap[target] }"></span>
            <span class="class-name">{{ classLabels[target] }}</span>
            <span class="class-count">{{ stats.targets[target].count }} 个</span>
            <span class="class-area">{{ formatArea(stats.targets[target].total_area_m2) }}</span>
          </div>
        </div>

        <button class="btn btn-sm btn-block" style="margin-top:10px" @click="zoomToResults">
          缩放至结果
        </button>
      </div>

      <!-- 导出 -->
      <div class="sidebar-section" v-if="stats">
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
          <input type="checkbox" v-model="showBaseMap" />
          <span style="color:#64748b">■</span> 底图
        </label>
        <label class="layer-toggle">
          <input type="checkbox" v-model="showTif" />
          <span style="color:#0891b2">■</span> 遥感影像
        </label>
        <label class="layer-toggle" v-if="resultGeojson">
          <input type="checkbox" v-model="showResults" />
          <span style="color:#10b981">■</span> 提取结果
        </label>
        <span v-if="tifInfo" style="margin-left:auto;color:var(--text-secondary);font-size:12px">
          {{ tifInfo.file_name }} | ~{{ tifInfo.resolution?.approx_meters }}m/px
        </span>
      </div>

      <MapView
        ref="mapRef"
        :tifInfo="tifInfo"
        :resultGeojson="resultGeojson"
        :showBaseMap="showBaseMap"
        :showTif="showTif"
        :showResults="showResults"
        :drawMode="drawMode"
        :resultColors="resultColors"
        @roi-drawn="handleRoiDrawn"
        @roi-cleared="handleRoiCleared"
      />

      <div class="status-bar">
        <span v-if="tifInfo">影像: {{ tifInfo.width }}x{{ tifInfo.height }}</span>
        <span v-if="roi">ROI: {{ ((roi.right - roi.left) * 111000).toFixed(0) }}m x {{ ((roi.top - roi.bottom) * 111000).toFixed(0) }}m</span>
        <span v-if="totalFeatures > 0">已提取: {{ totalFeatures }} 个要素</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import './assets/main.css'
import MapView from './components/MapView.vue'
import { getTifInfo, generateTiles, extractFeatures, exportResult } from './api/index.js'

// ── 状态 ──
const tifInfo = ref(null)
const mapRef = ref(null)
const drawMode = ref(false)
const useFullImage = ref(false)
const roi = ref(null)
const targets = ref(['building'])
const method = ref('geoai')
const extracting = ref(false)
const generatingTiles = ref(false)
const resultGeojson = ref(null)
const stats = ref(null)

const showBaseMap = ref(true)
const showTif = ref(true)
const showResults = ref(true)

const classLabels = { building: '建筑', forest: '林地', grassland: '草地' }
const classColorMap = { building: '#ef4444', forest: '#16a34a', grassland: '#84cc16' }
const resultColors = {
  building: { stroke: '#ef4444', fill: 'rgba(239,68,68,0.25)' },
  forest: { stroke: '#16a34a', fill: 'rgba(22,163,74,0.25)' },
  grassland: { stroke: '#84cc16', fill: 'rgba(132,204,22,0.25)' },
}

const params = reactive({
  // CV 参数
  building_canny_low: 50,
  building_canny_high: 150,
  building_min_area_px: 100,
  building_kernel_size: 5,
  forest_hsv_h_range: [35, 85],
  forest_hsv_s_min: 40,
  forest_hsv_v_min: 40,
  forest_min_area_px: 200,
  grassland_hsv_h_range: [25, 75],
  grassland_hsv_s_range: [20, 120],
  grassland_hsv_v_range: [60, 200],
  grassland_min_area_px: 300,
  simplify_tolerance: 1.0,
  min_polygon_area: 10,
  // GeoAI 参数
  geoai_device: 'cpu',
  geoai_building_regularize: true,
  geoai_building_batch_size: 4,
  geoai_groundedsam_threshold: 0.3,
})

const totalFeatures = computed(() => {
  if (!stats.value || !stats.value.targets) return 0
  return Object.values(stats.value.targets).reduce((sum, t) => sum + t.count, 0)
})

function formatArea(m2) {
  if (m2 > 1e6) return (m2 / 1e6).toFixed(2) + ' km²'
  if (m2 > 1e4) return (m2 / 1e4).toFixed(1) + ' 万m²'
  return m2.toFixed(0) + ' m²'
}

function toggleTarget(t) {
  const idx = targets.value.indexOf(t)
  if (idx >= 0) targets.value.splice(idx, 1)
  else targets.value.push(t)
}

function toggleDraw() {
  drawMode.value = !drawMode.value
  if (drawMode.value) useFullImage.value = false
}

function handleRoiDrawn(bounds) {
  roi.value = bounds
  drawMode.value = false
  useFullImage.value = false
}

function handleRoiCleared() {
  roi.value = null
}

function clearROI() {
  roi.value = null
  if (mapRef.value) mapRef.value.clearROI()
}

function zoomToResults() {
  if (mapRef.value) mapRef.value.zoomToResults()
}

// ── 初始化 ──
onMounted(async () => {
  try {
    tifInfo.value = await getTifInfo()
  } catch (e) {
    console.error('获取 TIF 信息失败', e)
  }
})

// ── 预生成瓦片 ──
async function handleGenTiles() {
  generatingTiles.value = true
  try {
    await generateTiles()
    alert('瓦片生成完成')
  } catch (e) {
    alert('瓦片生成失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    generatingTiles.value = false
  }
}

// ── 执行提取 ──
async function handleExtract() {
  if (targets.value.length === 0) {
    alert('请至少选择一个提取目标')
    return
  }

  extracting.value = true
  resultGeojson.value = null
  stats.value = null

  try {
    const config = {
      targets: targets.value,
      method: method.value,
      roi: useFullImage.value ? null : roi.value || null,
      ...params,
    }

    const res = await extractFeatures(config)

    // 合并所有目标的结果
    const allFeatures = []
    for (const target of targets.value) {
      const gj = res.results?.[target]
      if (gj && gj.features) {
        allFeatures.push(...gj.features)
      }
    }
    resultGeojson.value = { type: 'FeatureCollection', features: allFeatures }
    stats.value = res.stats
  } catch (e) {
    alert('提取失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    extracting.value = false
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
