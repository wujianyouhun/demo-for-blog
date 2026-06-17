<template>
  <div ref="mapEl" class="map-container"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import 'ol/ol.css'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import XYZ from 'ol/source/XYZ'
import GeoJSON from 'ol/format/GeoJSON'
import { Style, Stroke, Fill, Text as OlText, Circle as CircleStyle } from 'ol/style'
import { fromLonLat } from 'ol/proj'

const props = defineProps({
  rawGeojson: Object,
  issuesGeojson: Object,
  repairGeojson: Object,
  showRaw: { type: Boolean, default: true },
  showIssues: { type: Boolean, default: true },
  showRepair: { type: Boolean, default: true },
  highlightIssue: Object,
})

const mapEl = ref(null)
let map = null
let tileLayer = null
let rawLayer = null
let issueLayer = null
let repairLayer = null
let highlightLayer = null

// ── 样式 ──
const rawStyle = new Style({
  stroke: new Stroke({ color: 'rgba(100,116,139,0.6)', width: 1.5 }),
  fill: new Fill({ color: 'rgba(100,116,139,0.06)' }),
})

const severityColors = {
  HIGH: { stroke: '#ef4444', fill: 'rgba(239,68,68,0.15)' },
  MEDIUM: { stroke: '#f59e0b', fill: 'rgba(245,158,11,0.15)' },
  LOW: { stroke: '#6b7280', fill: 'rgba(107,114,128,0.12)' },
}

function issueStyleFn(feature) {
  const sev = feature.get('severity') || 'LOW'
  const colors = severityColors[sev] || severityColors.LOW
  return new Style({
    stroke: new Stroke({ color: colors.stroke, width: 2.5, lineDash: [6, 3] }),
    fill: new Fill({ color: colors.fill }),
    text: new OlText({
      text: feature.get('error_type') || '',
      font: 'bold 11px sans-serif',
      fill: new Fill({ color: colors.stroke }),
      offsetY: -14,
      backgroundFill: new Fill({ color: 'rgba(255,255,255,0.85)' }),
      padding: [2, 6, 2, 6],
    }),
  })
}

const repairStyle = new Style({
  stroke: new Stroke({ color: '#10b981', width: 2.5 }),
  fill: new Fill({ color: 'rgba(16,185,129,0.12)' }),
})

const highlightStyle = new Style({
  stroke: new Stroke({ color: '#f43f5e', width: 3.5 }),
  fill: new Fill({ color: 'rgba(244,63,94,0.2)' }),
})

function _addFeatures(layer, geojson) {
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
    map.getView().fit(extent, { size: map.getSize(), padding: [50, 50, 50, 50], duration: 500 })
  }
}

onMounted(() => {
  tileLayer = new TileLayer({
    source: new XYZ({
      url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      attributions: '&copy; OpenStreetMap',
    }),
    opacity: 0.4,
  })

  rawLayer = new VectorLayer({ source: new VectorSource(), style: rawStyle, zIndex: 5 })
  issueLayer = new VectorLayer({
    source: new VectorSource(),
    style: issueStyleFn,
    zIndex: 10,
  })
  repairLayer = new VectorLayer({ source: new VectorSource(), style: repairStyle, zIndex: 15 })
  highlightLayer = new VectorLayer({ source: new VectorSource(), style: highlightStyle, zIndex: 20 })

  map = new Map({
    target: mapEl.value,
    layers: [tileLayer, rawLayer, issueLayer, repairLayer, highlightLayer],
    view: new View({
      center: fromLonLat([108.945, 34.265]),
      zoom: 16,
    }),
  })
})

// 原始数据
watch(() => props.rawGeojson, (gj) => {
  if (!map) return
  rawLayer.setVisible(props.showRaw)
  _addFeatures(rawLayer, gj)
  if (gj && gj.features && gj.features.length > 0) {
    _fitExtent(gj)
  }
}, { deep: true })

// 问题标记
watch(() => props.issuesGeojson, (gj) => {
  if (!map) return
  issueLayer.setVisible(props.showIssues)
  _addFeatures(issueLayer, gj)
}, { deep: true })

// 修复结果
watch(() => props.repairGeojson, (gj) => {
  if (!map) return
  repairLayer.setVisible(props.showRepair)
  _addFeatures(repairLayer, gj)
}, { deep: true })

// 高亮单个问题
watch(() => props.highlightIssue, (issue) => {
  if (!map) return
  const src = highlightLayer.getSource()
  src.clear()
  if (!issue || !issue.geometry) return
  const gj = { type: 'FeatureCollection', features: [{ type: 'Feature', properties: issue, geometry: issue.geometry }] }
  _addFeatures(highlightLayer, gj)
}, { deep: true })

// 可见性
watch(() => props.showRaw, (v) => { if (rawLayer) rawLayer.setVisible(v) })
watch(() => props.showIssues, (v) => { if (issueLayer) issueLayer.setVisible(v) })
watch(() => props.showRepair, (v) => { if (repairLayer) repairLayer.setVisible(v) })
</script>
