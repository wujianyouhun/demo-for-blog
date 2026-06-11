<template>
  <div class="map-container" ref="mapEl"></div>
</template>

<script>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import Map from 'ol/Map'
import View from 'ol/View'
import Projection from 'ol/proj/Projection'
import ImageLayer from 'ol/layer/Image'
import ImageStatic from 'ol/source/ImageStatic'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import { fromExtent } from 'ol/geom/Polygon'
import { Style, Fill, Stroke, Circle as CircleStyle } from 'ol/style'
import { defaults as defaultInteractions, DoubleClickZoom, DragPan } from 'ol/interaction'

import {
  getDisplayImageUrl,
  predictByPoint,
  predictByBox,
  predictByText,
} from '../api/index.js'

export default {
  name: 'MapCanvas',
  props: {
    sessionId: String,
    mode: { type: String, default: 'pan' },
    promptText: { type: String, default: 'building' },
  },
  emits: ['loading', 'mask-updated', 'toast', 'cursor-move'],

  setup(props, { emit, expose }) {
    const mapEl = ref(null)
    let map = null
    let imageLayer = null
    let maskLayer = null
    let pointsLayer = null
    let boxLayer = null
    let dblClickZoomInteraction = null
    let dragPanInteraction = null

    // 影像显示尺寸（用于坐标映射）
    let displayWidth = 0
    let displayHeight = 0
    let imageExtent = [0, 0, 1, 1]  // [minX, minY, maxX, maxY]

    // 点标注状态
    let fgPoints = []   // [[x, y], ...] 像素坐标
    let bgPoints = []   // [[x, y], ...]

    // 框标注状态
    let isDrawingBox = false
    let boxStart = null

    // ── 初始化地图 ──
    function initMap() {
      // 自定义投影，像素坐标（左上角 0,0）
      const projection = new Projection({
        code: 'pixel',
        units: 'pixels',
        extent: [0, 0, 1, 1],  // 会在加载影像时更新
      })

      pointsLayer = new VectorLayer({
        source: new VectorSource(),
        style: (feature) => {
          const type = feature.get('type')
          return new Style({
            image: new CircleStyle({
              radius: 6,
              fill: new Fill({
                color: type === 'fg' ? '#2ecc71' : '#e74c3c',
              }),
              stroke: new Stroke({ color: '#fff', width: 2 }),
            }),
          })
        },
        zIndex: 20,
      })

      boxLayer = new VectorLayer({
        source: new VectorSource(),
        style: new Style({
          fill: new Fill({ color: 'rgba(52, 152, 219, 0.2)' }),
          stroke: new Stroke({ color: '#3498db', width: 2, lineDash: [6, 4] }),
        }),
        zIndex: 20,
      })

      // 创建可控制的双击缩放和拖拽平移
      dblClickZoomInteraction = new DoubleClickZoom()
      dragPanInteraction = new DragPan()

      map = new Map({
        target: mapEl.value,
        layers: [pointsLayer, boxLayer],
        view: new View({
          projection,
          center: [0, 0],
          zoom: 1,
          maxZoom: 10,
          minZoom: 0.1,
        }),
        controls: [],
        interactions: defaultInteractions({
          doubleClickZoom: false,
          dragPan: false,
        }),
      })

      // 手动添加可控的交互
      map.addInteraction(dblClickZoomInteraction)
      map.addInteraction(dragPanInteraction)

      // 初始禁用双击缩放（标注模式需要双击）
      dblClickZoomInteraction.setActive(false)
      dragPanInteraction.setActive(true)

      // 鼠标移动 → 报告像素坐标
      map.on('pointermove', (evt) => {
        if (!displayWidth) return
        const [x, y] = evt.coordinate
        // 转换到图像像素坐标（注意 Y 轴翻转）
        const px = Math.round(x)
        const py = Math.round(displayHeight - y)
        if (px >= 0 && px < displayWidth && py >= 0 && py < displayHeight) {
          emit('cursor-move', { x: px, y: py })
        }
      })

      // 点击事件
      map.on('singleclick', handleClick)
      map.on('dblclick', handleDoubleClick)

      // 框绘制事件
      map.on('pointerdown', handlePointerDown)
      map.on('pointerdrag', handlePointerDrag)
      map.on('pointerup', handlePointerUp)
    }

    // ── 加载影像到地图 ──
    function loadDisplayImage(sessionId, info) {
      displayWidth = info.display_width
      displayHeight = info.display_height

      // OpenLayers 使用左下角为原点的坐标系
      // 图像像素坐标 (0,0) 在左上角 → 映射到 OL 坐标 (0, displayHeight)
      imageExtent = [0, 0, displayWidth, displayHeight]

      // 更新投影范围
      const proj = map.getView().getProjection()
      proj.setExtent(imageExtent)

      // 添加影像图层
      const imgUrl = getDisplayImageUrl(sessionId)
      const imgSource = new ImageStatic({
        url: imgUrl,
        imageExtent: imageExtent,
        projection: proj,
      })

      if (imageLayer) {
        map.removeLayer(imageLayer)
      }
      imageLayer = new ImageLayer({
        source: imgSource,
        zIndex: 1,
      })
      map.addLayer(imageLayer)

      // 适配视图
      map.getView().fit(imageExtent, {
        size: map.getSize(),
        padding: [20, 20, 20, 20],
      })
    }

    // ── 获取点击的像素坐标 ──
    function getPixelCoord(evt) {
      const [x, y] = evt.coordinate
      const px = Math.round(x)
      const py = Math.round(displayHeight - y)
      return { px, py, ox: x, oy: y }
    }

    // ── 点击处理 ──
    function handleClick(evt) {
      if (!props.sessionId || props.mode === 'pan') return

      const { px, py } = getPixelCoord(evt)

      if (props.mode === 'point') {
        if (evt.originalEvent.shiftKey) {
          // Shift+Click = 背景点
          bgPoints.push([px, py])
          addPointMarker(px, py, 'bg')
        } else {
          // 普通点击 = 前景点
          fgPoints.push([px, py])
          addPointMarker(px, py, 'fg')
        }
      } else if (props.mode === 'text') {
        // 文本模式：直接调用文本分割
        doTextPredict(px, py)
      }
    }

    // ── 双击 = 执行点分割 ──
    function handleDoubleClick(evt) {
      if (props.mode !== 'point') return
      if (fgPoints.length === 0) return

      // 阻止双击缩放
      evt.preventDefault?.()

      doPointPredict()
    }

    // ── 框绘制 ──
    function handlePointerDown(evt) {
      if (props.mode !== 'box' || !props.sessionId) return
      isDrawingBox = true
      boxStart = getPixelCoord(evt)
      boxLayer.getSource().clear()
    }

    function handlePointerDrag(evt) {
      if (!isDrawingBox || props.mode !== 'box') return
      const end = getPixelCoord(evt)
      const source = boxLayer.getSource()
      source.clear()

      // 在 OL 坐标系中绘制矩形
      const x1 = Math.min(boxStart.px, end.px)
      const x2 = Math.max(boxStart.px, end.px)
      const y1 = displayHeight - Math.max(boxStart.py, end.py) // 翻转 Y
      const y2 = displayHeight - Math.min(boxStart.py, end.py)

      const boxFeature = new Feature({
        geometry: fromExtent([x1, y1, x2, y2]),
      })
      source.addFeature(boxFeature)
    }

    function handlePointerUp(evt) {
      if (!isDrawingBox || props.mode !== 'box') return
      isDrawingBox = false

      const end = getPixelCoord(evt)
      const x1 = Math.min(boxStart.px, end.px)
      const y1 = Math.min(boxStart.py, end.py)
      const x2 = Math.max(boxStart.px, end.px)
      const y2 = Math.max(boxStart.py, end.py)

      // 忽略太小的框
      if (x2 - x1 < 3 || y2 - y1 < 3) return

      doBoxPredict([x1, y1, x2, y2])
    }

    // ── 添加点标记 ──
    function addPointMarker(px, py, type) {
      const olY = displayHeight - py  // 翻转 Y 轴
      const feature = new Feature({
        geometry: new Point([px, olY]),
        type,
      })
      pointsLayer.getSource().addFeature(feature)
    }

    // ── API 调用 ──
    async function doPointPredict() {
      const allPoints = [...fgPoints, ...bgPoints]
      const allLabels = [
        ...fgPoints.map(() => 1),
        ...bgPoints.map(() => 0),
      ]

      emit('loading', true)
      try {
        const res = await predictByPoint(props.sessionId, allPoints, allLabels)
        const url = URL.createObjectURL(res.data)
        updateMaskOverlay(url)
        emit('mask-updated', true)
        emit('toast', '点分割完成', 'success')
        // 清除点标记
        clearPointMarkers()
      } catch (e) {
        emit('toast', '分割失败: ' + (e.response?.data?.detail || e.message), 'error')
      } finally {
        emit('loading', false)
      }
    }

    async function doBoxPredict(box) {
      emit('loading', true)
      try {
        const res = await predictByBox(props.sessionId, box)
        const url = URL.createObjectURL(res.data)
        updateMaskOverlay(url)
        emit('mask-updated', true)
        emit('toast', '框分割完成', 'success')
        boxLayer.getSource().clear()
      } catch (e) {
        emit('toast', '分割失败: ' + (e.response?.data?.detail || e.message), 'error')
      } finally {
        emit('loading', false)
      }
    }

    async function doTextPredict(px, py) {
      if (!props.promptText?.trim()) {
        emit('toast', '请输入目标文本', 'error')
        return
      }
      emit('loading', true)
      try {
        const res = await predictByText(props.sessionId, props.promptText)
        const url = URL.createObjectURL(res.data)
        updateMaskOverlay(url)
        emit('mask-updated', true)
        emit('toast', '文本分割完成', 'success')
      } catch (e) {
        emit('toast', '分割失败: ' + (e.response?.data?.detail || e.message), 'error')
      } finally {
        emit('loading', false)
      }
    }

    // ── Mask 叠加层 ──
    function updateMaskOverlay(pngUrl) {
      if (maskLayer) {
        map.removeLayer(maskLayer)
      }
      const maskSource = new ImageStatic({
        url: pngUrl,
        imageExtent: imageExtent,
        projection: map.getView().getProjection(),
      })
      maskLayer = new ImageLayer({
        source: maskSource,
        opacity: 0.6,
        zIndex: 10,
      })
      map.addLayer(maskLayer)
    }

    function clearMask() {
      if (maskLayer) {
        map.removeLayer(maskLayer)
        maskLayer = null
      }
      emit('mask-updated', false)
    }

    function clearPointMarkers() {
      fgPoints = []
      bgPoints = []
      pointsLayer.getSource().clear()
    }

    // ── 监听 mode 变化 → 更新光标与交互 ──
    watch(
      () => props.mode,
      (newMode) => {
        if (!map) return
        const el = mapEl.value

        if (newMode === 'pan') {
          el.style.cursor = 'grab'
          dblClickZoomInteraction?.setActive(true)
          dragPanInteraction?.setActive(true)
        } else if (newMode === 'point') {
          el.style.cursor = 'crosshair'
          dblClickZoomInteraction?.setActive(false)
          dragPanInteraction?.setActive(true)  // 允许平移+点击标注
        } else if (newMode === 'box') {
          el.style.cursor = 'crosshair'
          dblClickZoomInteraction?.setActive(false)
          dragPanInteraction?.setActive(false)  // 禁用平移以绘制框
        } else if (newMode === 'text') {
          el.style.cursor = 'text'
          dblClickZoomInteraction?.setActive(false)
          dragPanInteraction?.setActive(true)
        }

        // 模式切换时清除临时标记
        clearPointMarkers()
        boxLayer?.getSource()?.clear()
      }
    )

    // ── 暴露给父组件的方法 ──
    expose({
      updateMaskOverlay,
      clearMask,
      loadDisplayImage,
    })

    onMounted(() => {
      nextTick(() => {
        initMap()

        // 键盘事件监听
        const handleKeyDown = (e) => {
          if (!props.sessionId) return
          if (e.key === 'Enter' && props.mode === 'point' && fgPoints.length > 0) {
            e.preventDefault()
            doPointPredict()
          } else if (e.key === 'Escape') {
            clearPointMarkers()
            boxLayer?.getSource()?.clear()
          }
        }
        window.addEventListener('keydown', handleKeyDown)
        // 保存引用以便清理
        mapEl.value._keyHandler = handleKeyDown
      })
    })

    onUnmounted(() => {
      if (mapEl.value?._keyHandler) {
        window.removeEventListener('keydown', mapEl.value._keyHandler)
      }
      if (map) {
        map.setTarget(null)
        map = null
      }
    })

    return { mapEl }
  },
}
</script>

<style scoped>
.map-container {
  width: 100%;
  height: 100%;
  cursor: grab;
}
</style>
