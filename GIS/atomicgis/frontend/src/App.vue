<template>
  <div>
    <h1>AtomicGIS Sandbox</h1>
    <div id="map"></div>
  </div>
</template>

<script setup>
import maplibregl from 'maplibre-gl'
import { onMounted } from 'vue'

onMounted(() => {
  const map = new maplibregl.Map({
    container: 'map',
    style: 'https://demotiles.maplibre.org/style.json',
    center: [116.39, 39.9],
    zoom: 4
  })
})
</script>

<style>
#map {
  width: 100%;
  height: 600px;
}
</style>