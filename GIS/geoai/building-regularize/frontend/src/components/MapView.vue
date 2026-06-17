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
import { Style, Stroke, Fill, Text as OlText } from 'ol/style'
import { fromLonLat } from 'ol/proj'

const props = defineProps({
  rawGeojson: Object,
  stepGeojson: Object,
  showRaw: { type: Boolean, default: true },
  showResult: { type: Boolean, default: true },
})

const mapEl = ref(null)
let map = null
let tileLayer = null
let rawLayer = null
let stepLayer = null

// ── 样式 ──
const rawStyle = new Style({
  stroke: new Stroke({ color: '#ef4444', width: 2, lineDash: [4, 4] }),
  fill: new Fill({ color: 'rgba(239,68,68,0.08)' }),
})

const stepStyle = new Style({
  stroke: new Stroke({ color: '#2563eb', width: 2.5 }),
  fill: new Fill({ color: 'rgba(37,99,235,0.12)' }),
})

const finalStyle = (feature) => {
  const verts = feature.get('vertex_count') || 0
  return new Style({
    stroke: new Stroke({ color: '#10b981', width: 2.5 }),
    fill: new Fill({ color: 'rgba(16,185,129,0.15)' }),
    text: new OlText({
      text: `${verts}v`,
      font: '11px sans-serif',
      fill: new Fill({ color: '#1e293b' }),
      offsetY: -12,
    }),
  })
}

function _addFeatures(layer, geojson, style) {
  const src = layer.getSource()
  src.clear()
  if (!geojson || !geojson.features || geojson.features.length === 0) return
  const fmt = new GeoJSON()
  const features = fmt.readFeatures(geojson, { featureProjection: 'EPSG:3857' })
  src.addFeatures(features)
}

function _fitExtent(geojson) {
  if (!geojson || !geojson.features || geojson.features.length === 0) return
  const fmt = new GeoJSON()
  const features = fmt.readFeatures(geojson, { featureProjection: 'EPSG:3857' })
  const tmpSrc = new VectorSource({ features })
  const extent = tmpSrc.getExtent()
  if (extent && isFinite(extent[0])) {
    map.getView().fit(extent, { size: map.getSize(), padding: [40, 40, 40, 40], duration: 500 })
  }
}

onMounted(() => {
  // 天地图底图 (可选，国内可访问)
  tileLayer = new TileLayer({
    source: new XYZ({
      url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      attributions: '&copy; OpenStreetMap',
    }),
    opacity: 0.5,
  })

  rawLayer = new VectorLayer({ source: new VectorSource(), style: rawStyle, zIndex: 10 })
  stepLayer = new VectorLayer({
    source: new VectorSource(),
    style: (feature) => finalStyle(feature),
    zIndex: 20,
  })

  map = new Map({
    target: mapEl.value,
    layers: [tileLayer, rawLayer, stepLayer],
    view: new View({
      center: fromLonLat([108.945, 34.265]),
      zoom: 17,
    }),
  })
})

// 原始数据变化
watch(() => props.rawGeojson, (gj) => {
  if (!map) return
  rawLayer.setVisible(props.showRaw)
  _addFeatures(rawLayer, gj, rawStyle)
  if (gj && gj.features && gj.features.length > 0) {
    _fitExtent(gj)
  }
}, { deep: true })

// 步骤结果变化
watch(() => props.stepGeojson, (gj) => {
  if (!map) return
  stepLayer.setVisible(props.showResult)
  _addFeatures(stepLayer, gj, stepStyle)
}, { deep: true })

// 可见性开关
watch(() => props.showRaw, (v) => { if (rawLayer) rawLayer.setVisible(v) })
watch(() => props.showResult, (v) => { if (stepLayer) stepLayer.setVisible(v) })
</script>
