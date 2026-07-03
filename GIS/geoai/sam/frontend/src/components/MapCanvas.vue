<template>
  <div class="map-container" ref="mapEl"></div>
</template>

<script>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import XYZ from 'ol/source/XYZ'
import ImageLayer from 'ol/layer/Image'
import ImageStatic from 'ol/source/ImageStatic'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import { fromExtent } from 'ol/geom/Polygon'
import { Style, Fill, Stroke, Circle as CircleStyle } from 'ol/style'
import { defaults as defaultInteractions, DoubleClickZoom, DragPan } from 'ol/interaction'
import { fromLonLat, toLonLat } from 'ol/proj'
import { getCenter } from 'ol/extent'

import {
  predictByPoint,
  predictByBox,
  predictByText,
  getSessionProgress,
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
    let tileLayer = null
    let maskLayer = null
    let pointsLayer = null
    let boxLayer = null
    let dblClickZoomInteraction = null
    let dragPanInteraction = null

    // 影像地理范围 (EPSG:3857)
    let imageExtent3857 = null  // [minX, minY, maxX, maxY]

    // 点标注状态
    let fgPoints = []   // [[lon, lat], ...]
    let bgPoints = []
    let fgMarkers = []  // OL features for display
    let bgMarkers = []

    // 框标注状态
    let isDrawingBox = false
    let boxStart = null
    let lastBoxGeo = null

    // ── 初始化地图（标准 Web Mercator） ──
    function initMap() {
      pointsLayer = new VectorLayer({
        source: new VectorSource(),
        style: (feature) => {
          const type = feature.get('type')
          return new Style({
            image: new CircleStyle({
              radius: 6,
              fill: new Fill({ color: type === 'fg' ? '#2ecc71' : '#e74c3c' }),
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

      dblClickZoomInteraction = new DoubleClickZoom()
      dragPanInteraction = new DragPan()

      map = new Map({
        target: mapEl.value,
        layers: [pointsLayer, boxLayer],
        view: new View({
          center: [0, 0],
          zoom: 2,
          maxZoom: 22,
          minZoom: 1,
        }),
        controls: [],
        interactions: defaultInteractions({
          doubleClickZoom: false,
          dragPan: false,
        }),
      })

      map.addInteraction(dblClickZoomInteraction)
      map.addInteraction(dragPanInteraction)
      dblClickZoomInteraction.setActive(false)
      dragPanInteraction.setActive(true)

      // 鼠标移动 → 报告 WGS84 坐标
      map.on('pointermove', (evt) => {
        const [lon, lat] = toLonLat(evt.coordinate)
        emit('cursor-move', { x: lon.toFixed(6), y: lat.toFixed(6), lon, lat })
      })

      map.on('singleclick', handleClick)
      map.on('dblclick', handleDoubleClick)
      map.on('pointerdown', handlePointerDown)
      map.on('pointerdrag', handlePointerDrag)
      map.on('pointerup', handlePointerUp)
    }

    // ── 加载 TiTiler 瓦片图层 ──
    function loadDisplayImage(sessionId, info) {
      // 移除旧图层
      if (tileLayer) map.removeLayer(tileLayer)
      if (maskLayer) { map.removeLayer(maskLayer); maskLayer = null }

      // 创建 XYZ 瓦片源 (指向 TiTiler)
      const tileSource = new XYZ({
        url: info.tile_url,
        crossOrigin: 'anonymous',
        maxZoom: 22,
      })

      tileLayer = new TileLayer({
        source: tileSource,
        zIndex: 1,
      })
      map.addLayer(tileLayer)

      // 计算 EPSG:3857 范围用于 fit 视图和 mask 定位
      if (info.mask_extent) {
        imageExtent3857 = info.mask_extent
        map.getView().fit(imageExtent3857, {
          size: map.getSize(),
          padding: [30, 30, 30, 30],
        })
      } else if (info.bounds && (!info.crs || info.crs.toUpperCase() === 'EPSG:4326')) {
        const [left, bottom, right, top] = info.bounds
        const bl = fromLonLat([left, bottom])
        const tr = fromLonLat([right, top])
        imageExtent3857 = [bl[0], bl[1], tr[0], tr[1]]

        map.getView().fit(imageExtent3857, {
          size: map.getSize(),
          padding: [30, 30, 30, 30],
        })
      } else {
        emit('toast', '影像缺少可用的 EPSG:3857 范围，无法正确定位标注坐标', 'error')
      }
    }

    // ── 获取点击处的 WGS84 坐标 ──
    function getGeoCoord(evt) {
      const [lon, lat] = toLonLat(evt.coordinate)
      return { lon, lat, coord3857: evt.coordinate }
    }

    // ── 点击处理 ──
    function handleClick(evt) {
      if (!props.sessionId || props.mode === 'pan') return

      const { lon, lat, coord3857 } = getGeoCoord(evt)

      if (props.mode === 'point') {
        if (evt.originalEvent.shiftKey) {
          bgPoints.push([lon, lat])
          addPointMarker(coord3857, 'bg')
        } else {
          fgPoints.push([lon, lat])
          addPointMarker(coord3857, 'fg')
        }
      } else if (props.mode === 'text') {
        doTextPredict({ lon, lat })
      }
    }

    // ── 双击 = 执行点分割 ──
    function handleDoubleClick(evt) {
      if (props.mode !== 'point') return
      if (fgPoints.length === 0) return
      evt.preventDefault?.()
      doPointPredict()
    }

    // ── 框绘制 ──
    function handlePointerDown(evt) {
      if (props.mode !== 'box' || !props.sessionId) return
      isDrawingBox = true
      boxStart = evt.coordinate
      boxLayer.getSource().clear()
    }

    function handlePointerDrag(evt) {
      if (!isDrawingBox || props.mode !== 'box') return
      const source = boxLayer.getSource()
      source.clear()

      const x1 = Math.min(boxStart[0], evt.coordinate[0])
      const y1 = Math.min(boxStart[1], evt.coordinate[1])
      const x2 = Math.max(boxStart[0], evt.coordinate[0])
      const y2 = Math.max(boxStart[1], evt.coordinate[1])

      const boxFeature = new Feature({
        geometry: fromExtent([x1, y1, x2, y2]),
      })
      source.addFeature(boxFeature)
    }

    function handlePointerUp(evt) {
      if (!isDrawingBox || props.mode !== 'box') return
      isDrawingBox = false

      const end = evt.coordinate
      const x1 = Math.min(boxStart[0], end[0])
      const y1 = Math.min(boxStart[1], end[1])
      const x2 = Math.max(boxStart[0], end[0])
      const y2 = Math.max(boxStart[1], end[1])

      // 忽略太小的框 (EPSG:3857 米)
      if (x2 - x1 < 1 || y2 - y1 < 1) return

      // 转为 WGS84
      const [lon1, lat1] = toLonLat([x1, y1])
      const [lon2, lat2] = toLonLat([x2, y2])
      lastBoxGeo = [lon1, lat1, lon2, lat2]
      doBoxPredict(lastBoxGeo)
    }

    // ── 添加点标记 ──
    function addPointMarker(coord3857, type) {
      const feature = new Feature({
        geometry: new Point(coord3857),
        type,
      })
      pointsLayer.getSource().addFeature(feature)
    }

    // ── API 调用 ──
    async function getErrorMessage(e) {
      const data = e.response?.data
      if (data instanceof Blob) {
        try {
          const text = await data.text()
          const parsed = JSON.parse(text)
          return parsed.detail || text
        } catch (_) {
          return e.message
        }
      }
      return data?.detail || e.message
    }

    async function doPointPredict() {
      const allPoints = [...fgPoints, ...bgPoints]
      const allLabels = [
        ...fgPoints.map(() => 1),
        ...bgPoints.map(() => 0),
      ]

      emit('loading', { active: true, text: '点标注：正在调用 SAM 模型...' })
      try {
        const res = await predictByPoint(props.sessionId, allPoints, allLabels)
        const url = URL.createObjectURL(res.data)
        updateMaskOverlay(url)
        emit('mask-updated', true)
        emit('toast', '点分割完成', 'success')
        clearPointMarkers()
      } catch (e) {
        emit('toast', '分割失败: ' + await getErrorMessage(e), 'error')
      } finally {
        emit('loading', false)
      }
    }

    async function doBoxPredict(box) {
      emit('loading', { active: true, text: '框标注：正在调用 SAM 模型...' })
      try {
        const res = await predictByBox(props.sessionId, box)
        const url = URL.createObjectURL(res.data)
        updateMaskOverlay(url)
        emit('mask-updated', true)
        emit('toast', '框分割完成', 'success')
        boxLayer.getSource().clear()
      } catch (e) {
        emit('toast', '分割失败: ' + await getErrorMessage(e), 'error')
      } finally {
        emit('loading', false)
      }
    }

    async function doTextPredict(clickPoint = null) {
      if (!props.promptText?.trim()) {
        emit('toast', '请输入目标文本', 'error')
        return
      }
      emit('loading', { active: true, text: '文本标注：正在提交请求...' })
      let progressTimer = null
      const pollProgress = async () => {
        if (!props.sessionId) return
        try {
          const res = await getSessionProgress(props.sessionId)
          const op = res.data || {}
          if (op.name === 'predict_text' && op.message) {
            const percent = Math.round((op.progress || 0) * 100)
            emit('loading', {
              active: op.status !== 'completed' && op.status !== 'failed',
              text: `${op.message} (${percent}%)`,
            })
          }
        } catch (_) {
          // Keep the main request running even if progress polling misses once.
        }
      }
      progressTimer = window.setInterval(pollProgress, 1000)
      pollProgress()
      try {
        const res = await predictByText(props.sessionId, props.promptText, 0.25, 0.25, clickPoint)
        const url = URL.createObjectURL(res.data)
        updateMaskOverlay(url)
        emit('mask-updated', true)
        emit('toast', '文本分割完成', 'success')
      } catch (e) {
        emit('toast', '分割失败: ' + await getErrorMessage(e), 'error')
      } finally {
        if (progressTimer) {
          window.clearInterval(progressTimer)
        }
        emit('loading', false)
      }
    }

    // ── Mask 叠加层（使用 EPSG:3857 范围定位） ──
    function updateMaskOverlay(pngUrl) {
      if (maskLayer) map.removeLayer(maskLayer)

      if (!imageExtent3857) {
        emit('toast', '缺少影像地理范围，无法叠加 Mask', 'error')
        return
      }

      const maskSource = new ImageStatic({
        url: pngUrl,
        imageExtent: imageExtent3857,
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

    function getPromptPayload(mode) {
      if (mode === 'point') {
        return {
          points: [...fgPoints, ...bgPoints],
          labels: [
            ...fgPoints.map(() => 1),
            ...bgPoints.map(() => 0),
          ],
        }
      }
      if (mode === 'box') {
        return {
          boxes: lastBoxGeo ? [lastBoxGeo] : [],
        }
      }
      return {}
    }

    // ── 监听 mode 变化 ──
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
          dragPanInteraction?.setActive(true)
        } else if (newMode === 'box') {
          el.style.cursor = 'crosshair'
          dblClickZoomInteraction?.setActive(false)
          dragPanInteraction?.setActive(false)
        } else if (newMode === 'text') {
          el.style.cursor = 'text'
          dblClickZoomInteraction?.setActive(false)
          dragPanInteraction?.setActive(true)
        }

        clearPointMarkers()
        boxLayer?.getSource()?.clear()
      }
    )

    expose({
      updateMaskOverlay,
      clearMask,
      loadDisplayImage,
      getPromptPayload,
    })

    onMounted(() => {
      nextTick(() => {
        initMap()

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
