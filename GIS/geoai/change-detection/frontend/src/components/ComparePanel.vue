<template>
  <div class="panel-section">
    <h3 class="panel-title">对比分析</h3>

    <!-- 变化统计 -->
    <el-divider content-position="left">变化统计</el-divider>
    <el-row :gutter="12" v-if="stats">
      <el-col :span="12">
        <div class="stat-card">
          <div class="stat-value">{{ stats.change_ratio }}%</div>
          <div class="stat-label">变化比例</div>
        </div>
      </el-col>
      <el-col :span="12">
        <div class="stat-card">
          <div class="stat-value">{{ stats.changed_pixels }}</div>
          <div class="stat-label">变化像素</div>
        </div>
      </el-col>
    </el-row>
    <el-empty v-else description="暂无检测结果" :image-size="60" />

    <!-- 可视化 -->
    <el-divider content-position="left">可视化生成</el-divider>
    <el-form :model="visForm" label-width="80px" size="default">
      <el-form-item label="对比模式">
        <el-select v-model="visForm.mode">
          <el-option label="并列对比" value="side_by_side" />
          <el-option label="变化叠加" value="overlay" />
          <el-option label="差异热力图" value="heatmap" />
        </el-select>
      </el-form-item>
      <el-form-item label="变化颜色">
        <el-color-picker v-model="visForm.change_color" />
      </el-form-item>
      <el-form-item label="透明度">
        <el-slider v-model="visForm.opacity" :min="0" :max="1" :step="0.05" show-input />
      </el-form-item>
    </el-form>

    <el-button type="success" class="w-full" :loading="generating" @click="generateVis">
      <el-icon class="mr-1"><PictureFilled /></el-icon> 生成对比图
    </el-button>

    <!-- 矢量预览 -->
    <div class="mt-4" v-if="vectors.length">
      <el-divider content-position="left">变化区域</el-divider>
      <el-scrollbar max-height="180px">
        <div
          v-for="v in vectors" :key="v.path"
          class="vec-item"
          @click="previewVector(v.name)"
        >
          <span>{{ v.name }}</span>
          <span class="text-gray-400 text-xs">{{ fmtSize(v.size) }}</span>
        </div>
      </el-scrollbar>
    </div>

    <el-button size="small" text class="mt-2" @click="refreshResults">
      <el-icon><Refresh /></el-icon> 刷新结果
    </el-button>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api/client.js'

const props = defineProps({
  currentPair: { type: Object, default: null },
  detectResult: { type: Object, default: null },
})
const emit = defineEmits(['task', 'show-geojson'])

const generating = ref(false)
const stats = ref(null)
const vectors = ref([])

const visForm = reactive({
  mode: 'side_by_side',
  change_color: '#FF0000',
  opacity: 0.5,
})

watch(() => props.detectResult, async (r) => {
  if (r?.mask) {
    try {
      const res = await api.post('/api/compare/stats', { change_map_path: r.mask })
      stats.value = res.data
    } catch {}
  }
  refreshResults()
}, { immediate: true })

onMounted(() => refreshResults())

async function generateVis() {
  if (!props.currentPair) { ElMessage.warning('请先下载数据'); return }
  generating.value = true
  try {
    const payload = {
      image_a: props.currentPair.time_a,
      image_b: props.currentPair.time_b,
      change_map: props.detectResult?.mask || null,
      mode: visForm.mode,
      change_color: visForm.change_color,
      opacity: visForm.opacity,
    }
    const r = await api.post('/api/compare/visualize', payload)
    ElMessage.success(`对比图已生成: ${r.data.filename}`)
    emit('task', { status: 'success', message: '对比图生成完成' })
  } catch (e) { ElMessage.error(e.response?.data?.detail || e.message) }
  finally { generating.value = false }
}

async function refreshResults() {
  try { const r = await api.get('/api/detect/results'); vectors.value = r.data.vectors || [] } catch {}
}

async function previewVector(filename) {
  try {
    const r = await api.get(`/api/compare/preview/${filename}`)
    if (r.data.geojson) emit('show-geojson', r.data.geojson)
  } catch { ElMessage.error('预览失败') }
}

function fmtSize(b) {
  if (!b) return '0 B'
  const u = ['B', 'KB', 'MB']; let i = 0
  while (b >= 1024 && i < 2) { b /= 1024; i++ }
  return b.toFixed(1) + ' ' + u[i]
}
</script>

<style scoped>
.vec-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 0; border-bottom: 1px solid #f5f5f5; cursor: pointer;
  font-size: 13px;
}
.vec-item:hover { color: #409eff; }
</style>
