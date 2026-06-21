<template>
  <div ref="mapContainer" class="map-container"></div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import L from 'leaflet'

const emit = defineEmits(['map-click', 'layer-added'])

const mapContainer = ref(null)
let map = null
let geoJsonLayer = null

const CLASS_COLORS = {
  0: '#808080', // background - gray
  1: '#e6194b', // building - red
  2: '#ffe119', // road - yellow
  3: '#3cb44b', // water - blue/green
  4: '#42d4f4', // vegetation - cyan
  5: '#f58231', // barren - orange
}

const CLASS_NAMES = {
  0: '背景',
  1: '建筑',
  2: '道路',
  3: '水体',
  4: '植被',
  5: '裸地',
}

onMounted(() => {
  map = L.map(mapContainer.value, {
    center: [39.9, 116.4],
    zoom: 10,
    zoomControl: true,
  })

  const streetLayer = L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }
  )

  const satelliteLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: '&copy; Esri', maxZoom: 18 }
  )

  streetLayer.addTo(map)

  const baseMaps = {
    '街道地图': streetLayer,
    '卫星影像': satelliteLayer,
  }

  L.control.layers(baseMaps).addTo(map)

  map.on('click', (e) => {
    emit('map-click', { lat: e.latlng.lat, lng: e.latlng.lng })
  })
})

onBeforeUnmount(() => {
  if (map) {
    map.remove()
    map = null
  }
})

function getStyle(feature) {
  const classId = feature.properties?.class_id ?? 0
  const color = CLASS_COLORS[classId] || '#808080'
  return {
    fillColor: color,
    weight: 1.5,
    opacity: 0.8,
    color: '#333',
    fillOpacity: 0.5,
  }
}

function onEachFeature(feature, layer) {
  const classId = feature.properties?.class_id ?? 0
  const className = CLASS_NAMES[classId] || `类别${classId}`
  const area = feature.properties?.area ? `${feature.properties.area.toFixed(2)} m²` : '-'
  layer.bindPopup(`
    <div style="min-width:140px">
      <b>${className}</b><br/>
      类别ID: ${classId}<br/>
      面积: ${area}
    </div>
  `)
}

function addGeoJSON(geojson) {
  clearLayers()
  geoJsonLayer = L.geoJSON(geojson, {
    style: getStyle,
    onEachFeature: onEachFeature,
  }).addTo(map)

  if (geoJsonLayer.getBounds().isValid()) {
    map.fitBounds(geoJsonLayer.getBounds(), { padding: [30, 30] })
  }

  emit('layer-added')
}

function clearLayers() {
  if (geoJsonLayer) {
    map.removeLayer(geoJsonLayer)
    geoJsonLayer = null
  }
}

defineExpose({ addGeoJSON, clearLayers })
</script>

<style scoped>
.map-container {
  width: 100%;
  height: 100%;
  min-height: 500px;
  border-radius: 4px;
}
</style>
