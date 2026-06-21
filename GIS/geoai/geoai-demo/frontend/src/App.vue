<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="header-left">
        <h1>GeoAI Demo - 土地覆盖分类系统</h1>
      </div>
      <div class="header-right">
        <el-tag type="success" size="small">DeepLabV3+</el-tag>
        <el-tag type="info" size="small">Semantic Segmentation</el-tag>
      </div>
    </el-header>

    <el-main class="app-main">
      <el-row :gutter="16" style="height: 100%">
        <el-col :span="10" class="panel-col">
          <el-card class="panel-card">
            <el-tabs v-model="activeTab" type="border-card">
              <el-tab-pane label="数据管理" name="data">
                <DataPanel @status-change="onStatusChange" />
              </el-tab-pane>
              <el-tab-pane label="模型训练" name="train">
                <TrainPanel @status-change="onStatusChange" />
              </el-tab-pane>
              <el-tab-pane label="要素提取" name="extract">
                <ExtractPanel @show-result="onShowResult" @status-change="onStatusChange" />
              </el-tab-pane>
              <el-tab-pane label="要素正则化" name="regularize">
                <RegularizePanel @show-result="onShowResult" @status-change="onStatusChange" />
              </el-tab-pane>
            </el-tabs>
          </el-card>
        </el-col>

        <el-col :span="14" class="map-col">
          <el-card class="map-card">
            <template #header>
              <div class="map-header">
                <span>地图预览</span>
                <el-button size="small" @click="clearMap">清除图层</el-button>
              </div>
            </template>
            <MapView ref="mapRef" />
          </el-card>
        </el-col>
      </el-row>
    </el-main>

    <el-footer class="app-footer">
      <span v-if="statusMessage">{{ statusMessage }}</span>
      <span v-else>就绪</span>
    </el-footer>
  </el-container>
</template>

<script setup>
import { ref } from 'vue'
import DataPanel from './components/DataPanel.vue'
import TrainPanel from './components/TrainPanel.vue'
import ExtractPanel from './components/ExtractPanel.vue'
import RegularizePanel from './components/RegularizePanel.vue'
import MapView from './components/MapView.vue'

const activeTab = ref('data')
const mapRef = ref(null)
const statusMessage = ref('')

function onShowResult(geojson) {
  if (mapRef.value) {
    mapRef.value.addGeoJSON(geojson)
  }
}

function onStatusChange(msg) {
  statusMessage.value = msg
}

function clearMap() {
  if (mapRef.value) {
    mapRef.value.clearLayers()
  }
}
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #1a73e8, #0d47a1);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.header-left h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.header-right {
  display: flex;
  gap: 8px;
}

.app-main {
  flex: 1;
  padding: 16px;
  overflow: hidden;
}

.panel-col,
.map-col {
  height: 100%;
}

.panel-card,
.map-card {
  height: 100%;
}

.panel-card :deep(.el-card__body) {
  padding: 0;
  height: calc(100% - 20px);
  overflow-y: auto;
}

.map-card :deep(.el-card__body) {
  padding: 0;
  height: calc(100% - 50px);
}

.map-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

.app-footer {
  background: #f5f7fa;
  border-top: 1px solid #e4e7ed;
  font-size: 13px;
  color: #606266;
  display: flex;
  align-items: center;
  padding: 0 24px;
  height: 36px;
}
</style>
