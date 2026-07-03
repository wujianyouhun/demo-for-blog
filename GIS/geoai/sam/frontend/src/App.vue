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
                     :placeholder="dataDir ? dataDir + '/image.tif' : '例如: F:/data/image.tif'" />
            </div>
            <div class="form-row" v-if="availableImages.length">
              <label>data 目录</label>
              <select v-model="imagePath">
                <option v-for="path in availableImages" :key="path" :value="path">
                  {{ fileName(path) }}
                </option>
              </select>
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

        <!-- 整幅影像处理 -->
        <div class="panel-section" v-if="sessionId">
          <div class="panel-title">整幅影像处理</div>
          <div class="form-row">
            <label>处理模式</label>
            <select v-model="fullMode">
              <option value="text">文本</option>
              <option value="auto">自动</option>
              <option value="point">点提示</option>
              <option value="box">框提示</option>
            </select>
          </div>
          <div class="form-row" v-if="fullMode === 'text'">
            <label>目标文本</label>
            <input class="input-field" type="text" v-model="promptText"
                   placeholder="building, road, water..." />
          </div>
          <div class="form-row">
            <label>瓦片尺寸</label>
            <input type="number" v-model.number="fullTileSize" min="512" step="256" />
          </div>
          <div class="form-row">
            <label>重叠像素</label>
            <input type="number" v-model.number="fullOverlap" min="0" step="64" />
          </div>
          <div class="form-row">
            <label>最小面积</label>
            <input type="number" v-model.number="fullMinArea" min="0" step="10" />
          </div>
          <div class="form-row">
            <label>格式</label>
            <select v-model="fullOutputFormat">
              <option value="gpkg">GeoPackage</option>
              <option value="geojson">GeoJSON</option>
              <option value="shp">Shapefile</option>
            </select>
          </div>
          <button class="btn btn-primary" style="width:100%; margin-top:4px;"
                  :disabled="loading || fullTaskStatus === 'running'"
                  @click="handleStartFullProcess">
            启动整图处理
          </button>
          <div v-if="fullTaskId" style="font-size:11px; color:var(--text-secondary); margin-top:6px;">
            <p>任务: {{ fullTaskStatus || '-' }} | {{ Math.round(fullProgress * 100) }}%</p>
            <progress style="width:100%; height:10px;" max="1" :value="fullProgress"></progress>
            <p v-if="fullTotalTiles">瓦片: {{ fullDoneTiles }} / {{ fullTotalTiles }}</p>
            <p>{{ fullMessage }}</p>
            <p v-if="fullError" style="color:#ff8a9a">{{ fullError }}</p>
            <p v-if="fullResult">多边形数: {{ fullResult.polygon_count }}</p>
          </div>
          <button class="btn" style="width:100%; margin-top:4px;"
                  :disabled="fullTaskStatus !== 'completed'"
                  @click="handleDownloadFullResult">
            下载整图结果
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

        <!-- 后台信息 -->
        <div class="panel-section">
          <div class="panel-title">后台信息</div>
          <div class="backend-log">
            <p v-if="!backendLogs.length">暂无后台信息</p>
            <p v-for="(log, i) in backendLogs" :key="i" :class="'log-' + log.level">
              <span>[{{ log.time }}]</span> {{ log.message }}
            </p>
          </div>
        </div>
      </aside>

      <!-- 地图主区域 -->
      <main class="app-main">
        <MapCanvas
          ref="mapCanvas"
          :session-id="sessionId"
          :mode="mode"
          :prompt-text="promptText"
          @loading="handleLoading"
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
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import MapCanvas from './components/MapCanvas.vue'
import {
  getConfig, loadImage, getImageInfo, postprocessMask,
  exportVectors, downloadExport, clearSession,
  startFullProcess, getFullProcessStatus, downloadFullProcessResult,
  getBackendLogs,
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
    const backendLogs = ref([])
    let backendLogTimer = null
    const dataDir = ref('')
    const availableImages = ref([])
    const fullMode = ref('text')
    const fullTileSize = ref(2048)
    const fullOverlap = ref(256)
    const fullMinArea = ref(50)
    const fullOutputFormat = ref('gpkg')
    const fullTaskId = ref('')
    const fullTaskStatus = ref('')
    const fullProgress = ref(0)
    const fullDoneTiles = ref(0)
    const fullTotalTiles = ref(0)
    const fullMessage = ref('')
    const fullError = ref('')
    const fullResult = ref(null)

    // ── 影像加载参数 ──
    const imagePath = ref('')
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

    function fileName(path) {
      return path.split(/[\\/]/).pop() || path
    }

    function handleLoading(payload) {
      if (typeof payload === 'boolean') {
        loading.value = payload
        if (!payload) loadingText.value = ''
        return
      }
      loading.value = !!payload?.active
      loadingText.value = payload?.text || ''
    }

    async function refreshBackendLogs() {
      try {
        const res = await getBackendLogs(40)
        backendLogs.value = (res.data.logs || []).slice(-8).reverse()
      } catch (_) {
        // 后端启动中或暂不可用时保持当前面板状态。
      }
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

    function validateFullProcessPayload(payload) {
      if (fullMode.value === 'point' && (!payload.points || payload.points.length === 0)) {
        addToast('点提示整图处理前，请先在地图上添加前景点/背景点', 'error')
        return false
      }
      if (fullMode.value === 'box' && (!payload.boxes || payload.boxes.length === 0)) {
        addToast('框提示整图处理前，请先拖拽一个框', 'error')
        return false
      }
      if (fullMode.value === 'text' && !promptText.value.trim()) {
        addToast('请输入目标文本', 'error')
        return false
      }
      return true
    }

    async function pollFullProcess(taskId) {
      try {
        const res = await getFullProcessStatus(taskId)
        const task = res.data
        fullTaskStatus.value = task.status
        fullProgress.value = task.progress || 0
        fullDoneTiles.value = task.done_tiles || 0
        fullTotalTiles.value = task.total_tiles || 0
        fullMessage.value = task.message || ''
        fullError.value = task.error || ''
        fullResult.value = task.result || null

        if (task.status === 'completed') {
          addToast('整图处理完成', 'success')
          return
        }
        if (task.status === 'failed') {
          addToast('整图处理失败: ' + (task.error || task.message), 'error', 6000)
          return
        }
        setTimeout(() => pollFullProcess(taskId), 2000)
      } catch (e) {
        addToast('查询整图任务失败: ' + (e.response?.data?.detail || e.message), 'error')
      }
    }

    async function handleStartFullProcess() {
      const promptPayload = mapCanvas.value?.getPromptPayload(fullMode.value) || {}
      if (!validateFullProcessPayload(promptPayload)) return

      loading.value = true
      loadingText.value = '提交整图任务中...'
      fullTaskStatus.value = ''
      fullProgress.value = 0
      fullDoneTiles.value = 0
      fullTotalTiles.value = 0
      fullMessage.value = ''
      fullError.value = ''
      fullResult.value = null

      try {
        const payload = {
          session_id: sessionId.value,
          mode: fullMode.value,
          text: promptText.value,
          tile_size: fullTileSize.value,
          overlap: fullOverlap.value,
          min_area: fullMinArea.value,
          output_format: fullOutputFormat.value,
          postprocess: true,
          ...promptPayload,
        }
        const res = await startFullProcess(payload)
        fullTaskId.value = res.data.task_id
        fullTaskStatus.value = res.data.status
        addToast('整图处理任务已启动', 'success')
        pollFullProcess(fullTaskId.value)
      } catch (e) {
        addToast('启动整图处理失败: ' + (e.response?.data?.detail || e.message), 'error', 6000)
      } finally {
        loading.value = false
      }
    }

    async function handleDownloadFullResult() {
      try {
        const res = await downloadFullProcessResult(fullTaskId.value)
        const blob = res.data
        const ext = fullOutputFormat.value === 'shp' ? '.zip'
                  : fullOutputFormat.value === 'geojson' ? '.geojson' : '.gpkg'
        const a = document.createElement('a')
        a.href = URL.createObjectURL(blob)
        a.download = `full_image_result${ext}`
        a.click()
        addToast('整图结果下载已开始', 'success')
      } catch (e) {
        addToast('下载整图结果失败: ' + (e.response?.data?.detail || e.message), 'error')
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
        dataDir.value = res.data.data_dir || ''
        availableImages.value = res.data.available_images || []
        imagePath.value = res.data.default_image || imagePath.value
        if (!imagePath.value && availableImages.value.length) {
          imagePath.value = availableImages.value[0]
        }
        modelType.value = res.data.default_model_type || modelType.value
        samVersion.value = res.data.default_sam_version || samVersion.value
      } catch (e) {
        // Keep the built-in defaults if the API is not ready yet.
      }
      refreshBackendLogs()
      backendLogTimer = window.setInterval(refreshBackendLogs, 3000)
    })

    onUnmounted(() => {
      if (backendLogTimer) {
        window.clearInterval(backendLogTimer)
      }
    })

    return {
      sessionId, imageInfo, loading, loadingText, mode, hasMask,
      cursorPos, toasts, exportResult, mapCanvas, backendLogs,
      dataDir, availableImages,
      fullMode, fullTileSize, fullOverlap, fullMinArea, fullOutputFormat,
      fullTaskId, fullTaskStatus, fullProgress, fullDoneTiles, fullTotalTiles,
      fullMessage, fullError, fullResult,
      imagePath, modelType, samVersion,
      promptText,
      ppMinSize, ppFillHoles, ppSmoothSigma, ppOpeningRadius, ppClosingRadius,
      exportMinArea, exportFormat,
      sessionStatus, modeHint,
      addToast, fileName, handleLoading, handleLoadImage, handlePostProcess,
      handleExport, handleDownload, handleClear,
      handleStartFullProcess, handleDownloadFullResult,
      handlePointExecute, handleClearPoints,
    }
  },
}
</script>
