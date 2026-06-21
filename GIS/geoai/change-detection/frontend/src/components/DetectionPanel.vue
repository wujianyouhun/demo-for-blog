<template>
  <div class="panel-section">
    <h3 class="panel-title">变化检测</h3>

    <!-- 模型训练 -->
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
      <el-form-item label="时相 A">
        <el-input v-model="detectForm.image_a" placeholder="选择或输入路径" />
      </el-form-item>
      <el-form-item label="时相 B">
        <el-input v-model="detectForm.image_b" placeholder="选择或输入路径" />
      </el-form-item>
      <el-form-item label="模型">
        <el-select v-model="detectForm.model_path" placeholder="选择模型" filterable>
          <el-option v-for="m in models" :key="m.path" :label="m.name" :value="m.path" />
        </el-select>
      </el-form-item>
      <el-form-item label="模型架构">
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
const result = ref(null)
const models = ref([])

const trainForm = reactive({ model_name: 'siamese_unet', epochs: 50, batch_size: 8, learning_rate: 0.0001 })
const detectForm = reactive({
  image_a: '', image_b: '', model_path: '', model_name: 'siamese_unet',
  tile_size: 256, overlap: 32, threshold: 0.5, smoothing_sigma: 1.0, min_area: 30,
})

watch(() => props.currentPair, (p) => {
  if (p) { detectForm.image_a = p.time_a; detectForm.image_b = p.time_b }
}, { immediate: true })

onMounted(() => refreshModels())

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
  if (!detectForm.image_a || !detectForm.image_b || !detectForm.model_path) {
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

function poll(taskId) {
  const iv = setInterval(async () => {
    try {
      const r = await api.get(`/api/detect/status/${taskId}`)
      if (r.data.status === 'completed') {
        clearInterval(iv)
        result.value = r.data
        ElMessage.success(r.data.message)
        emit('task', { status: 'success', message: r.data.message })
        if (r.data.result) emit('result', r.data.result)
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
  try { const r = await api.get('/api/detect/models'); models.value = r.data.models || [] } catch {}
}
</script>
