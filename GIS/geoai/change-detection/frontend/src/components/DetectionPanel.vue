<template>
  <div class="panel-section">
    <h3 class="panel-title">变化检测</h3>

    <!-- 模型训练 -->
    <el-divider content-position="left">样本制作</el-divider>
    <div class="sample-status">
      <span>样本 A: {{ sampleCounts.time_a }}</span>
      <span>样本 B: {{ sampleCounts.time_b }}</span>
      <span>标签: {{ sampleCounts.labels }}</span>
    </div>
    <el-form :model="sampleForm" label-width="80px" size="default">
      <el-form-item label="模式">
        <el-select v-model="sampleForm.mode">
          <el-option label="模拟样本" value="synthetic" />
          <el-option label="GeoAI 弱标签" value="weak-label" />
          <el-option label="矢量标签" value="vector-label" />
        </el-select>
      </el-form-item>
      <el-form-item label="样本数" v-if="sampleForm.mode === 'synthetic'">
        <el-input-number v-model="sampleForm.num_samples" :min="2" :max="2000" />
      </el-form-item>
      <el-form-item label="矢量标签" v-if="sampleForm.mode === 'vector-label'">
        <el-input v-model="sampleForm.vector_label" placeholder="变化矢量 GeoJSON/GPKG/SHP 路径" />
      </el-form-item>
      <el-form-item label="切片大小" v-if="sampleForm.mode !== 'synthetic'">
        <el-input-number v-model="sampleForm.tile_size" :min="128" :max="1024" :step="64" />
      </el-form-item>
    </el-form>
    <el-button type="success" class="w-full" :loading="sampling" @click="startSample">
      <el-icon class="mr-1"><Files /></el-icon> 制作训练样本
    </el-button>

    <el-divider content-position="left">模型训练</el-divider>
    <el-form :model="trainForm" label-width="80px" size="default">
      <el-form-item label="模型">
        <el-select v-model="trainForm.model_name">
          <el-option label="Siamese U-Net (推荐)" value="siamese_unet" />
          <el-option label="BiT Transformer" value="bit" />
        </el-select>
      </el-form-item>
      <el-form-item label="训练轮数">
        <el-input-number v-model="trainForm.epochs" :min="1" :max="200" />
      </el-form-item>
      <el-form-item label="批大小">
        <el-input-number v-model="trainForm.batch_size" :min="1" :max="32" />
      </el-form-item>
      <el-form-item label="学习率">
        <el-input-number v-model="trainForm.learning_rate" :min="0.00001" :max="0.01" :step="0.00001" :precision="5" />
      </el-form-item>
    </el-form>

    <el-button type="warning" class="w-full" :loading="training" @click="startTrain">
      <el-icon class="mr-1"><TrendCharts /></el-icon> 训练模型
    </el-button>

    <!-- 推理检测 -->
    <el-divider content-position="left">推理检测</el-divider>
    <el-form :model="detectForm" label-width="80px" size="default">
      <el-form-item label="引擎">
        <el-segmented v-model="detectForm.engine" :options="engineOptions" />
      </el-form-item>
      <el-form-item label="时相 A">
        <el-input v-model="detectForm.image_a" placeholder="选择或输入路径" />
      </el-form-item>
      <el-form-item label="时相 B">
        <el-input v-model="detectForm.image_b" placeholder="选择或输入路径" />
      </el-form-item>
      <el-form-item label="GeoAI模型" v-if="detectForm.engine === 'geoai'">
        <el-select v-model="detectForm.model_name" placeholder="选择 GeoAI ChangeStar 模型" filterable>
          <el-option v-for="m in geoaiModels" :key="m.name" :label="m.name" :value="m.name" />
        </el-select>
      </el-form-item>
      <el-form-item label="模型权重" v-if="detectForm.engine === 'cdd'">
        <el-select v-model="detectForm.model_path" placeholder="选择模型" filterable>
          <el-option v-for="m in models" :key="m.path" :label="m.name" :value="m.path" />
        </el-select>
      </el-form-item>
      <el-form-item label="模型架构" v-if="detectForm.engine === 'cdd'">
        <el-select v-model="detectForm.model_name">
          <el-option label="Siamese U-Net" value="siamese_unet" />
          <el-option label="BiT Transformer" value="bit" />
        </el-select>
      </el-form-item>
      <el-form-item label="阈值">
        <el-slider v-model="detectForm.threshold" :min="0.1" :max="0.9" :step="0.05" show-input />
      </el-form-item>
      <el-form-item label="平滑">
        <el-slider v-model="detectForm.smoothing_sigma" :min="0" :max="5" :step="0.1" show-input />
      </el-form-item>
      <el-form-item label="最小面积">
        <el-input-number v-model="detectForm.min_area" :min="0" :max="1000" :step="10" />
      </el-form-item>
    </el-form>

    <el-button type="primary" class="w-full" :loading="detecting" @click="startDetect">
      <el-icon class="mr-1"><Search /></el-icon> 执行变化检测
    </el-button>

    <!-- 检测结果 -->
    <div class="mt-3" v-if="result">
      <el-alert :title="result.message" :type="result.status === 'completed' ? 'success' : 'error'" show-icon :closable="false" />
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api/client.js'

const props = defineProps({ currentPair: { type: Object, default: null } })
const emit = defineEmits(['task', 'result'])

const training = ref(false)
const detecting = ref(false)
const sampling = ref(false)
const result = ref(null)
const models = ref([])
const geoaiModels = ref([])
const sampleCounts = reactive({ time_a: 0, time_b: 0, labels: 0 })

const engineOptions = [
  { label: 'GeoAI', value: 'geoai' },
  { label: '自训练', value: 'cdd' },
]

const trainForm = reactive({ model_name: 'siamese_unet', epochs: 50, batch_size: 8, learning_rate: 0.0001 })
const sampleForm = reactive({
  mode: 'synthetic',
  num_samples: 100,
  tile_size: 256,
  stride: 256,
  min_change_pixels: 20,
  vector_label: '',
  geoai_model: 's1_s1c1_vitb',
  overwrite: true,
})
const detectForm = reactive({
  engine: 'geoai',
  image_a: '', image_b: '', model_path: '', model_name: 's1_s1c1_vitb',
  tile_size: 1024, overlap: 64, threshold: 0.5, smoothing_sigma: 1.0, min_area: 30,
})

watch(() => props.currentPair, (p) => {
  if (p) { detectForm.image_a = p.time_a; detectForm.image_b = p.time_b }
}, { immediate: true })

watch(() => detectForm.engine, (engine) => {
  if (engine === 'geoai') {
    detectForm.model_name = 's1_s1c1_vitb'
    detectForm.tile_size = 1024
    detectForm.overlap = 64
  } else {
    detectForm.model_name = 'siamese_unet'
    detectForm.tile_size = 256
    detectForm.overlap = 32
  }
})

onMounted(() => { refreshModels(); refreshSamples() })

async function startSample() {
  if (sampleForm.mode !== 'synthetic' && (!detectForm.image_a || !detectForm.image_b)) {
    ElMessage.warning('请先选择或填写双时相影像路径'); return
  }
  sampling.value = true
  try {
    const payload = {
      ...sampleForm,
      image_a: detectForm.image_a || null,
      image_b: detectForm.image_b || null,
      vector_label: sampleForm.vector_label || null,
    }
    const r = await api.post('/api/detect/samples', payload)
    ElMessage.success('样本制作已启动')
    emit('task', { status: 'running', message: `样本任务 ${r.data.task_id}` })
    poll(r.data.task_id, 'sample')
  } catch (e) { ElMessage.error(e.response?.data?.detail || e.message) }
  finally { sampling.value = false }
}

async function startTrain() {
  training.value = true
  try {
    const r = await api.post('/api/detect/train', trainForm)
    ElMessage.success('训练已启动')
    emit('task', { status: 'running', message: `训练任务 ${r.data.task_id}` })
    poll(r.data.task_id)
  } catch (e) { ElMessage.error(e.response?.data?.detail || e.message) }
  finally { training.value = false }
}

async function startDetect() {
  if (!detectForm.image_a || !detectForm.image_b || (detectForm.engine === 'cdd' && !detectForm.model_path)) {
    ElMessage.warning('请填写完整的检测参数'); return
  }
  detecting.value = true; result.value = null
  try {
    const r = await api.post('/api/detect/run', detectForm)
    ElMessage.success('检测任务已启动')
    emit('task', { status: 'running', message: '变化检测运行中...' })
    poll(r.data.task_id)
  } catch (e) { ElMessage.error(e.response?.data?.detail || e.message) }
  finally { detecting.value = false }
}

function poll(taskId, kind = 'detect') {
  const iv = setInterval(async () => {
    try {
      const r = await api.get(`/api/detect/status/${taskId}`)
      if (r.data.status === 'completed') {
        clearInterval(iv)
        result.value = r.data
        ElMessage.success(r.data.message)
        emit('task', { status: 'success', message: r.data.message })
        if (kind === 'sample') refreshSamples()
        if (kind === 'detect' && r.data.result) emit('result', r.data.result)
        refreshModels()
      } else if (r.data.status === 'failed') {
        clearInterval(iv)
        result.value = r.data
        ElMessage.error(r.data.message)
        emit('task', { status: 'error', message: r.data.message })
      }
    } catch {}
  }, 5000)
}

async function refreshModels() {
  try {
    const r = await api.get('/api/detect/models')
    models.value = r.data.models || []
    geoaiModels.value = r.data.geoai_models || [{ name: 's1_s1c1_vitb' }]
  } catch {}
}

async function refreshSamples() {
  try {
    const r = await api.get('/api/detect/samples')
    Object.assign(sampleCounts, r.data.counts || {})
  } catch {}
}
</script>

<style scoped>
.sample-status {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-bottom: 12px;
  color: #606266;
  font-size: 12px;
}
</style>
