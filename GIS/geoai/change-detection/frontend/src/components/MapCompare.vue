<template>
  <div class="map-wrapper">
    <!-- 主地图 -->
    <div ref="mapEl" class="ol-map"></div>

    <!-- 卷帘分隔线 -->
    <div
      v-if="compareMode === 'swipe' && hasPair"
      ref="swipeEl"
      class="swipe-handle"
      :style="{ left: swipePos + '%' }"
      @mousedown="startSwipe"
    >
      <div class="swipe-line"></div>
      <div class="swipe-circle">
        <span class="swipe-arrows">◀ ▶</span>
      </div>
    </div>

    <!-- 标签 -->
    <div class="map-label left-label" v-if="compareMode === 'swipe' && hasPair">时相 A</div>
    <div class="map-label right-label" v-if="compareMode === 'swipe' && hasPair">时相 B</div>

    <!-- 变化图叠加 (change 模式) -->
    <div class="map-label change-label" v-if="compareMode === 'change' && changeMapUrl">
      变化检测结果
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import ImageLayer from 'ol/layer/Image'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import OSM from 'ol/source/OSM'
import ImageStatic from 'ol/source/ImageStatic'
import GeoJSON from 'ol/format/GeoJSON'
import { fromLonLat, transformExtent } from 'ol/proj'
import { Style, Fill, Stroke } from 'ol/style'

const props = defineProps({
  compareMode: { type: String, default: 'swipe' },
  geojsonData: { type: Object, default: null },
  imageUrlA: { type: String, default: null },
  imageUrlB: { type: String, default: null },
  changeMapUrl: { type: String, default: null },
  imageBounds: { type: Array, default: null }, // [west, south, east, north] in EPSG:4326
})

const mapEl = ref(null)
const swipeEl = ref(null)
const swipePos = ref(50)
const hasPair = ref(false)

let map = null
let baseLayer = null
let layerA = null
let layerB = null
let changeLayer = null
let vectorLayer = null

// ── Swipe handlers (attached to layerB) ──
let preRenderHandler = null
let postRenderHandler = null

function attachSwipeToLayer(layer) {
  detachSwipeFromLayer()

  preRenderHandler = (e) => {
    if (props.compareMode !== 'swipe') return
    const ctx = e.context
    if (!ctx) return
    const width = ctx.canvas.width
    const height = ctx.canvas.height
    const clipX = width * swipePos.value / 100
    ctx.save()
    ctx.beginPath()
    ctx.rect(clipX, 0, width - clipX, height)
    ctx.clip()
  }

  postRenderHandler = (e) => {
    if (props.compareMode !== 'swipe') return
    const ctx = e.context
    if (!ctx) return
    ctx.restore()
  }

  layer.on('prerender', preRenderHandler)
  layer.on('postrender', postRenderHandler)
}

function detachSwipeFromLayer() {
  if (layerB && preRenderHandler) {
    layerB.un('prerender', preRenderHandler)
    layerB.un('postrender', postRenderHandler)
  }
  preRenderHandler = null
  postRenderHandler = null
}

// ── Map init ──
onMounted(() => {
  baseLayer = new TileLayer({ source: new OSM() })

  map = new Map({
    target: mapEl.value,
    layers: [baseLayer],
    view: new View({
      center: fromLonLat([116.4, 39.9]),
      zoom: 10,
    }),
  })
})

onUnmounted(() => {
  detachSwipeFromLayer()
  if (map) {
    map.setTarget(null)
    map = null
  }
})

// ── Build an ImageStatic layer from a preview URL ──
function createImageLayer(url, bounds) {
  if (!url || !bounds) return null
  const extent = transformExtent(bounds, 'EPSG:4326', 'EPSG:3857')
  return new ImageLayer({
    source: new ImageStatic({
      url,
      imageExtent: extent,
    }),
    opacity: 1,
  })
}

// ── Load image pair onto the map ──
function loadImagePair(urlA, urlB, bounds) {
  removeLayerA()
  removeLayerB()
  removeChangeLayer()

  if (!urlA && !urlB) {
    hasPair.value = false
    return
  }

  // Layer A: always visible (base imagery)
  if (urlA && bounds) {
    layerA = createImageLayer(urlA, bounds)
    if (layerA) map.addLayer(layerA)
  }

  // Layer B: clipped by swipe in swipe mode
  if (urlB && bounds) {
    layerB = createImageLayer(urlB, bounds)
    if (layerB) {
      map.addLayer(layerB)
      attachSwipeToLayer(layerB)
    }
  }

  hasPair.value = !!(urlA && urlB)

  // Zoom to extent
  if (bounds) {
    const extent = transformExtent(bounds, 'EPSG:4326', 'EPSG:3857')
    map.getView().fit(extent, { padding: [40, 40, 40, 40] })
  }
}

// ── Compare mode changes ──
watch(() => props.compareMode, (mode) => {
  // Handle layer visibility based on mode
  if (mode === 'change') {
    // Show change map, hide A/B
    if (layerA) layerA.setVisible(false)
    if (layerB) layerB.setVisible(false)
    showChangeLayer()
  } else if (mode === 'side') {
    // Both visible, no clipping
    if (layerA) layerA.setVisible(true)
    if (layerB) { layerB.setVisible(true); layerB.setOpacity(0.5) }
    removeChangeLayer()
  } else {
    // swipe mode: both visible, layerB clipped
    if (layerA) layerA.setVisible(true)
    if (layerB) { layerB.setVisible(true); layerB.setOpacity(1) }
    removeChangeLayer()
  }
  if (map) map.render()
})

// ── Watch image URLs from parent ──
watch(
  () => [props.imageUrlA, props.imageUrlB, props.imageBounds],
  ([urlA, urlB, bounds]) => {
    if (urlA || urlB) {
      nextTick(() => loadImagePair(urlA, urlB, bounds))
    }
  },
)

// ── Watch change map ──
watch(() => props.changeMapUrl, (url) => {
  if (url && props.compareMode === 'change') {
    showChangeLayer()
  }
})

function showChangeLayer() {
  removeChangeLayer()
  if (!props.changeMapUrl || !props.imageBounds) return
  const extent = transformExtent(props.imageBounds, 'EPSG:4326', 'EPSG:3857')
  changeLayer = new ImageLayer({
    source: new ImageStatic({
      url: props.changeMapUrl,
      imageExtent: extent,
    }),
    opacity: 0.7,
  })
  map.addLayer(changeLayer)
}

// ── Watch GeoJSON vector data ──
watch(() => props.geojsonData, (gj) => {
  if (!gj || !map) return
  removeVectorLayer()
  const source = new VectorSource({
    features: new GeoJSON().readFeatures(gj, { featureProjection: 'EPSG:3857' }),
  })
  vectorLayer = new VectorLayer({
    source,
    style: new Style({
      fill: new Fill({ color: 'rgba(255, 0, 0, 0.3)' }),
      stroke: new Stroke({ color: '#ff0000', width: 2 }),
    }),
  })
  map.addLayer(vectorLayer)
  if (source.getFeatures().length > 0) {
    map.getView().fit(source.getExtent(), { padding: [40, 40, 40, 40] })
  }
})

// ── Swipe drag ──
function startSwipe(e) {
  e.preventDefault()
  const startX = e.clientX
  const startPos = swipePos.value
  const mapWidth = mapEl.value.clientWidth

  const onMove = (ev) => {
    const dx = ev.clientX - startX
    swipePos.value = Math.max(0, Math.min(100, startPos + dx / mapWidth * 100))
    if (map) map.render()
  }
  const onUp = () => {
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
  }
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}

// ── Cleanup helpers ──
function removeLayerA() {
  if (layerA) { map.removeLayer(layerA); layerA = null }
}
function removeLayerB() {
  detachSwipeFromLayer()
  if (layerB) { map.removeLayer(layerB); layerB = null }
}
function removeChangeLayer() {
  if (changeLayer) { map.removeLayer(changeLayer); changeLayer = null }
}
function removeVectorLayer() {
  if (vectorLayer) { map.removeLayer(vectorLayer); vectorLayer = null }
}

defineExpose({ map, loadImagePair })
</script>

<style scoped>
.map-wrapper { position: relative; width: 100%; height: 100%; }
.ol-map { width: 100%; height: 100%; }

.map-label {
  position: absolute; top: 12px; padding: 4px 12px;
  background: rgba(0,0,0,0.6); color: #fff; border-radius: 4px;
  font-size: 12px; z-index: 100; pointer-events: none;
}
.left-label { left: 12px; }
.right-label { right: 12px; }
.change-label { left: 50%; transform: translateX(-50%); }

/* 卷帘手柄 */
.swipe-handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 4px;
  cursor: col-resize;
  z-index: 200;
  transform: translateX(-50%);
}
.swipe-line {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 3px;
  background: #fff;
  box-shadow: 0 0 6px rgba(0,0,0,0.5);
  transform: translateX(-50%);
}
.swipe-circle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  display: flex;
  align-items: center;
  justify-content: center;
}
.swipe-arrows {
  font-size: 10px;
  color: #333;
  white-space: nowrap;
  user-select: none;
}
</style>
