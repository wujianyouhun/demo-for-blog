<template>
  <div ref="mapEl" class="map-container"></div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import 'ol/ol.css'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import XYZ from 'ol/source/XYZ'
import GeoJSON from 'ol/format/GeoJSON'
import Draw from 'ol/interaction/Draw'
import { Style, Stroke, Fill, Text as OlText } from 'ol/style'
import { fromLonLat, toLonLat } from 'ol/proj'
import { getArea } from 'ol/sphere'

const props = defineProps({
  tifInfo: Object,
  resultGeojson: Object,
  showBaseMap: { type: Boolean, default: true },
  showTif: { type: Boolean, default: true },
  showResults: { type: Boolean, default: true },
  drawMode: { type: Boolean, default: false },
  resultColors: { type: Object, default: () => ({}) },
})

const emit = defineEmits(['roi-drawn', 'roi-cleared'])

const mapEl = ref(null)
let map = null
let osmLayer = null
let tifLayer = null
let resultLayer = null
let drawInteraction = null
let roiLayer = null

// 类别默认颜色
const classColors = {
  building: { stroke: '#ef4444', fill: 'rgba(239,68,68,0.25)' },
  forest: { stroke: '#16a34a', fill: 'rgba(22,163,74,0.25)' },
  grassland: { stroke: '#84cc16', fill: 'rgba(132,204,22,0.25)' },
}

function resultStyleFn(feature) {
  const cls = feature.get('class') || 'building'
  const colors = props.resultColors[cls] || classColors[cls] || classColors.building
  const area = feature.get('area_m2') || 0
  return new Style({
    stroke: new Stroke({ color: colors.stroke, width: 2 }),
    fill: new Fill({ color: colors.fill }),
    text: area > 50 ? new OlText({
      text: `${Math.round(area)}m²`,
      font: '11px sans-serif',
      fill: new Fill({ color: colors.stroke }),
      offsetY: -10,
    }) : undefined,
  })
}

const roiStyle = new Style({
  stroke: new Stroke({ color: '#3b82f6', width: 2.5, lineDash: [8, 4] }),
  fill: new Fill({ color: 'rgba(59,130,246,0.08)' }),
})

onMounted(() => {
  osmLayer = new TileLayer({
    source: new XYZ({
      url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      attributions: '&copy; OpenStreetMap',
    }),
    opacity: 0.6,
  })

  tifLayer = new TileLayer({
    source: new XYZ({ url: '/api/tiles/{z}/{x}/{y}' }),
    opacity: 1.0,
    zIndex: 5,
  })

  roiLayer = new VectorLayer({
    source: new VectorSource(),
    style: roiStyle,
    zIndex: 10,
  })

  resultLayer = new VectorLayer({
    source: new VectorSource(),
    style: resultStyleFn,
    zIndex: 20,
  })

  map = new Map({
    target: mapEl.value,
    layers: [osmLayer, tifLayer, roiLayer, resultLayer],
    view: new View({
      center: fromLonLat([108.96, 34.19]),
      zoom: 14,
    }),
  })
})

// TIF 范围 → 定位
watch(() => props.tifInfo, (info) => {
  if (!map || !info || !info.bounds) return
  const b = info.bounds
  const extent = [
    ...fromLonLat([b.left, b.bottom]),
    ...fromLonLat([b.right, b.top]),
  ]
  map.getView().fit(extent, { size: map.getSize(), padding: [40, 40, 40, 40], duration: 600 })
}, { deep: true })

// 结果显示
watch(() => props.resultGeojson, (gj) => {
  if (!map) return
  const src = resultLayer.getSource()
  src.clear()
  resultLayer.setVisible(props.showResults)
  if (!gj || !gj.features || gj.features.length === 0) return
  const fmt = new GeoJSON()
  const features = fmt.readFeatures(gj, { featureProjection: 'EPSG:3857' })
  src.addFeatures(features)
}, { deep: true })

// 绘制模式
watch(() => props.drawMode, (enabled) => {
  if (!map) return
  if (drawInteraction) {
    map.removeInteraction(drawInteraction)
    drawInteraction = null
  }
  if (enabled) {
    drawInteraction = new Draw({
      source: roiLayer.getSource(),
      type: 'Polygon',
      maxPoints: 2,  // 矩形: 两点确定
    })
    drawInteraction.on('drawend', (evt) => {
      const geom = evt.feature.getGeometry()
      const extent = geom.getExtent()
      const bottomLeft = toLonLat([extent[0], extent[1]])
      const topRight = toLonLat([extent[2], extent[3]])
      emit('roi-drawn', {
        left: bottomLeft[0],
        bottom: bottomLeft[1],
        right: topRight[0],
        top: topRight[1],
      })
    })
    map.addInteraction(drawInteraction)
  }
})

// 图层可见性
watch(() => props.showBaseMap, (v) => { if (osmLayer) osmLayer.setVisible(v) })
watch(() => props.showTif, (v) => { if (tifLayer) tifLayer.setVisible(v) })
watch(() => props.showResults, (v) => { if (resultLayer) resultLayer.setVisible(v) })

// 缩放至结果
function zoomToResults() {
  const src = resultLayer.getSource()
  if (src.getFeatures().length === 0) return
  const extent = src.getExtent()
  if (extent && isFinite(extent[0])) {
    map.getView().fit(extent, { size: map.getSize(), padding: [60, 60, 60, 60], duration: 500 })
  }
}

function clearROI() {
  roiLayer.getSource().clear()
  emit('roi-cleared')
}

defineExpose({ zoomToResults, clearROI })
</script>
