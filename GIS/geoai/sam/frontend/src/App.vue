<template>
  <div class="app-layout">
    <!-- 顶部标题栏 -->
    <header class="app-header">
      <h1>SAM GeoAI 标注平台</h1>
      <span style="flex:1"></span>
      <span style="font-size:12px; color:var(--text-secondary)">
        {{ sessionStatus }}
      </span>
    </header>

    <div class="app-body">
      <!-- 左侧控制面板 -->
      <aside class="app-sidebar">
        <!-- 影像加载 -->
        <div class="panel-section">
          <div class="panel-title">影像文件</div>
          <div v-if="!sessionId" style="margin-top:4px;">
            <div class="form-row">
              <label>文件路径</label>
              <input class="input-field" type="text" v-model="imagePath"
                     placeholder="例如: F:/data/image.tif" />
            </div>
            <div class="form-row">
              <label>模型</label>
              <select v-model="modelType">
                <option value="vit_b">vit_b (轻量)</option>
                <option value="vit_l">vit_l (推荐)</option>
                <option value="vit_h">vit_h (精细)</option>
              </select>
            </div>
            <div class="form-row">
              <label>SAM 版本</label>
              <select v-model="samVersion">
                <option value="sam1">SAM 1</option>
                <option value="sam2">SAM 2</option>
              </select>
            </div>
            <button class="btn btn-primary" style="width:100%; margin-top:8px;"
                    :disabled="loading || !imagePath"
                    @click="handleLoadImage">
              加载影像
            </button>
          </div>
          <div v-else style="font-size:12px; color:var(--text-secondary)">
            <p>{{ imageInfo?.filename || '已加载' }}</p>
            <p>{{ imageInfo?.width }} x {{ imageInfo?.height }} 像素</p>
            <p v-if="imageInfo?.crs">CRS: {{ imageInfo.crs }}</p>
            <p v-if="imageInfo?.bounds">
              范围: {{ imageInfo.bounds[0].toFixed(4) }}, {{ imageInfo.bounds[1].toFixed(4) }}
              ~ {{ imageInfo.bounds[2].toFixed(4) }}, {{ imageInfo.bounds[3].toFixed(4) }}
            </p>
          </div>
        </div>

        <!-- 标注工具栏 -->
        <div class="panel-section" v-if="sessionId">
          <div class="panel-title">标注模式</div>
          <div class="btn-group">
            <button class="btn" :class="{ active: mode === 'point' }"
                    @click="mode = 'point'" title="点标注：左键前景，右键背景">
              点标注
            </button>
            <button class="btn" :class="{ active: mode === 'box' }"
                    @click="mode = 'box'" title="框标注：拖拽绘制矩形">
              框标注
            </button>
            <button class="btn" :class="{ active: mode === 'text' }"
                    @click="mode = 'text'" title="文本标注：输入目标名称">
              文本标注
            </button>
            <button class="btn" :class="{ active: mode === 'pan' }"
                    @click="mode = 'pan'" title="平移/缩放">
              平移
            </button>
          </div>
          <div v-if="mode === 'text'" style="margin-top:8px;">
            <div class="form-row">
              <label>目标文本</label>
              <input class="input-field" type="text" v-model="promptText"
                     placeholder="building, tree, road..." />
            </div>
          </div>
          <p style="font-size:11px; color:var(--text-secondary); margin-top:6px;">
            {{ modeHint }}
          </p>
          <div v-if="mode === 'point'" style="margin-top:8px;">
            <button class="btn btn-primary" style="width:100%;"
                    :disabled="loading"
                    @click="handlePointExecute">
              执行点分割 (Enter)
            </button>
            <button class="btn" style="width:100%; margin-top:4px;"
                    @click="handleClearPoints">
              清除标记点 (Esc)
            </button>
          </div>
        </div>

        <!-- 后处理 -->
        <div class="panel-section" v-if="sessionId">
          <div class="panel-title">Mask 后处理</div>
          <div class="form-row">
            <label>最小面积</label>
            <input type="number" v-model.number="ppMinSize" min="0" step="50" />
          </div>
          <div class="form-row">
            <label>填充孔洞</label>
            <input type="checkbox" v-model="ppFillHoles" />
          </div>
          <div class="form-row">
            <label>平滑 Sigma</label>
            <input type="number" v-model.number="ppSmoothSigma" min="0" step="0.5" />
          </div>
          <div class="form-row">
            <label>开运算半径</label>
            <input type="number" v-model.number="ppOpeningRadius" min="0" step="1" />
          </div>
          <div class="form-row">
            <label>闭运算半径</label>
            <input type="number" v-model.number="ppClosingRadius" min="0" step="1" />
          </div>
          <button class="btn" style="width:100%; margin-top:4px;"
                  :disabled="!hasMask || loading"
                  @click="handlePostProcess">
            执行后处理
          </button>
        </div>

        <!-- 矢量导出 -->
        <div class="panel-section" v-if="sessionId">
          <div class="panel-title">矢量导出</div>
          <div class="form-row">
            <label>最小面积</label>
            <input type="number" v-model.number="exportMinArea" min="0" step="10" />
          </div>
          <div class="form-row">
            <label>导出格式</label>
            <select v-model="exportFormat">
              <option value="geojson">GeoJSON</option>
              <option value="gpkg">GeoPackage</option>
              <option value="shp">Shapefile</option>
            </select>
          </div>
          <div class="btn-group" style="margin-top:4px;">
            <button class="btn btn-primary" :disabled="!hasMask || loading"
                    @click="handleExport">
              导出矢量
            </button>
            <button class="btn" :disabled="!exportResult?.file" @click="handleDownload">
              下载文件
            </button>
          </div>
          <p v-if="exportResult" style="font-size:11px; color:var(--text-secondary); margin-top:4px;">
            多边形数: {{ exportResult.polygon_count ?? '-' }}
            <span v-if="exportResult.file"> | {{ exportResult.file }}</span>
          </p>
        </div>

        <!-- 清除 -->
        <div class="panel-section" v-if="sessionId">
          <button class="btn" style="width:100%;" @click="handleClear">
            清除当前 Mask
          </button>
        </div>
      </aside>

      <!-- 地图主区域 -->
      <main class="app-main">
        <MapCanvas
          ref="mapCanvas"
          :session-id="sessionId"
          :mode="mode"
          :prompt-text="promptText"
          @loading="loading = $event"
          @mask-updated="hasMask = $event"
          @toast="addToast"
          @cursor-move="cursorPos = $event"
        />

        <!-- Loading 遮罩 -->
        <div class="loading-overlay" v-if="loading">
          <div class="spinner"></div>
          <span>{{ loadingText }}</span>
        </div>
      </main>
    </div>

    <!-- 底部状态栏 -->
    <footer class="app-statusbar">
      <span v-if="cursorPos">经纬度: {{ cursorPos.lon }}, {{ cursorPos.lat }}</span>
      <span v-if="imageInfo">原图: {{ imageInfo.width }} x {{ imageInfo.height }}</span>
      <span v-if="imageInfo?.crs">CRS: {{ imageInfo.crs }}</span>
      <span v-if="hasMask">Mask 已生成</span>
      <span style="flex:1"></span>
      <span v-if="loading">{{ loadingText }}</span>
    </footer>

    <!-- Toast 通知 -->
    <div class="toast-container">
      <div v-for="(t, i) in toasts" :key="i" :class="'toast toast-' + t.type">
        {{ t.msg }}
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, nextTick, onMounted } from 'vue'
import MapCanvas from './components/MapCanvas.vue'
import {
  getConfig, loadImage, getImageInfo, postprocessMask,
  exportVectors, downloadExport, clearSession
} from './api/index.js'

export default {
  name: 'App',
  components: { MapCanvas },

  setup() {
    // ── 状态 ──
    const sessionId = ref(null)
    const imageInfo = ref(null)
    const loading = ref(false)
    const loadingText = ref('')
    const mode = ref('pan')
    const hasMask = ref(false)
    const cursorPos = ref(null)
    const toasts = ref([])
    const exportResult = ref(null)
    const mapCanvas = ref(null)

    // ── 影像加载参数 ──
    const imagePath = ref('D:\\西安19级.tif')
    const modelType = ref('vit_l')
    const samVersion = ref('sam1')

    // ── 文本标注 ──
    const promptText = ref('building')

    // ── 后处理参数 ──
    const ppMinSize = ref(200)
    const ppFillHoles = ref(true)
    const ppSmoothSigma = ref(1.5)
    const ppOpeningRadius = ref(2)
    const ppClosingRadius = ref(3)

    // ── 导出参数 ──
    const exportMinArea = ref(50)
    const exportFormat = ref('geojson')

    const sessionStatus = computed(() => {
      if (!sessionId.value) return '未加载影像'
      return `会话: ${sessionId.value.slice(0, 8)}...`
    })

    const modeHint = computed(() => {
      switch (mode.value) {
        case 'point': return '左键标记前景，Shift+左键标记背景，双击或 Enter 执行分割'
        case 'box': return '拖拽绘制矩形框，松开后自动分割'
        case 'text': return '点击影像任意位置，使用文本提示进行分割'
        case 'pan': return '拖拽平移，滚轮缩放（支持无级缩放到像素级）'
        default: return ''
      }
    })

    // ── 方法 ──
    function addToast(msg, type = 'info', duration = 3000) {
      const idx = toasts.value.length
      toasts.value.push({ msg, type })
      setTimeout(() => {
        toasts.value.splice(idx, 1)
      }, duration)
    }

    async function handleLoadImage() {
      loading.value = true
      loadingText.value = '加载影像中...'
      try {
        const res = await loadImage(imagePath.value, modelType.value, samVersion.value)
        sessionId.value = res.data.session_id
        imageInfo.value = res.data
        mode.value = 'point'
        addToast('影像加载成功', 'success')
        // 通知地图组件加载显示影像
        await nextTick()
        mapCanvas.value?.loadDisplayImage(res.data.session_id, res.data)
      } catch (e) {
        addToast('加载失败: ' + (e.response?.data?.detail || e.message), 'error')
      } finally {
        loading.value = false
      }
    }

    async function handlePostProcess() {
      loading.value = true
      loadingText.value = '后处理中...'
      try {
        const res = await postprocessMask(sessionId.value, {
          min_size: ppMinSize.value,
          fill_holes: ppFillHoles.value,
          smooth_sigma: ppSmoothSigma.value,
          opening_radius: ppOpeningRadius.value,
          closing_radius: ppClosingRadius.value,
        })
        // 更新 mask 叠加层
        const url = URL.createObjectURL(res.data)
        mapCanvas.value?.updateMaskOverlay(url)
        addToast('后处理完成', 'success')
      } catch (e) {
        addToast('后处理失败: ' + (e.response?.data?.detail || e.message), 'error')
      } finally {
        loading.value = false
      }
    }

    async function handleExport() {
      loading.value = true
      loadingText.value = '矢量化导出中...'
      try {
        const res = await exportVectors(
          sessionId.value,
          exportMinArea.value,
          exportFormat.value,
        )
        exportResult.value = res.data
        addToast(`导出成功，${res.data.polygon_count} 个多边形`, 'success')
      } catch (e) {
        addToast('导出失败: ' + (e.response?.data?.detail || e.message), 'error')
      } finally {
        loading.value = false
      }
    }

    async function handleDownload() {
      try {
        const res = await downloadExport(sessionId.value)
        const blob = res.data
        const ext = exportFormat.value === 'geojson' ? '.geojson'
                  : exportFormat.value === 'gpkg' ? '.gpkg' : '.shp'
        const a = document.createElement('a')
        a.href = URL.createObjectURL(blob)
        a.download = `polygons${ext}`
        a.click()
        addToast('下载已开始', 'success')
      } catch (e) {
        addToast('下载失败', 'error')
      }
    }

    async function handleClear() {
      try {
        await clearSession(sessionId.value)
        hasMask.value = false
        mapCanvas.value?.clearMask()
        addToast('Mask 已清除', 'info')
      } catch (e) {
        addToast('清除失败', 'error')
      }
    }

    function handlePointExecute() {
      // 触发 MapCanvas 内的点分割逻辑（通过模拟 Enter 键）
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }))
    }

    function handleClearPoints() {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    }

    onMounted(async () => {
      try {
        const res = await getConfig()
        imagePath.value = res.data.default_image || imagePath.value
        modelType.value = res.data.default_model_type || modelType.value
        samVersion.value = res.data.default_sam_version || samVersion.value
      } catch (e) {
        // Keep the built-in defaults if the API is not ready yet.
      }
    })

    return {
      sessionId, imageInfo, loading, loadingText, mode, hasMask,
      cursorPos, toasts, exportResult, mapCanvas,
      imagePath, modelType, samVersion,
      promptText,
      ppMinSize, ppFillHoles, ppSmoothSigma, ppOpeningRadius, ppClosingRadius,
      exportMinArea, exportFormat,
      sessionStatus, modeHint,
      addToast, handleLoadImage, handlePostProcess,
      handleExport, handleDownload, handleClear,
      handlePointExecute, handleClearPoints,
    }
  },
}
</script>
