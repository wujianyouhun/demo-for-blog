<template>
  <div class="panel-content">
    <h3>模型训练</h3>

    <el-form label-width="90px" size="default">
      <el-form-item label="模型">
        <el-select v-model="model" placeholder="选择模型" style="width: 100%">
          <el-option label="DeepLabV3+ ResNet50" value="deeplabv3p_resnet50" />
          <el-option label="DeepLabV3+ ResNet101" value="deeplabv3p_resnet101" />
          <el-option label="DeepLabV3+ Xception" value="deeplabv3p_xception" />
          <el-option label="DeepLabV3+ MobileNetV2" value="deeplabv3p_mobilenetv2" />
        </el-select>
      </el-form-item>

      <el-form-item label="训练轮数">
        <el-input-number v-model="epochs" :min="1" :max="500" :step="5" style="width: 100%" />
      </el-form-item>

      <el-form-item label="批量大小">
        <el-input-number v-model="batchSize" :min="1" :max="64" :step="1" style="width: 100%" />
      </el-form-item>

      <el-form-item label="学习率">
        <el-input-number
          v-model="learningRate"
          :min="0.00001"
          :max="0.1"
          :step="0.0001"
          :precision="5"
          style="width: 100%"
        />
      </el-form-item>

      <el-form-item>
        <el-button type="warning" :loading="preparing" @click="prepareSamples">
          准备样本
        </el-button>
        <el-button type="primary" :loading="training" :disabled="!samplesReady" @click="startTraining">
          开始训练
        </el-button>
      </el-form-item>
    </el-form>

    <el-divider>训练进度</el-divider>

    <el-descriptions :column="2" border size="small" v-if="trainingInfo.epoch > 0">
      <el-descriptions-item label="当前轮次">
        {{ trainingInfo.epoch }} / {{ epochs }}
      </el-descriptions-item>
      <el-descriptions-item label="当前损失">
        {{ trainingInfo.loss.toFixed(4) }}
      </el-descriptions-item>
      <el-descriptions-item label="mIoU">
        {{ (trainingInfo.miou * 100).toFixed(2) }}%
      </el-descriptions-item>
      <el-descriptions-item label="状态">
        <el-tag :type="trainingInfo.status === 'running' ? 'warning' : 'success'" size="small">
          {{ trainingInfo.status === 'running' ? '训练中' : '已完成' }}
        </el-tag>
      </el-descriptions-item>
    </el-descriptions>

    <el-progress
      v-if="training"
      :percentage="Math.round((trainingInfo.epoch / epochs) * 100)"
      :stroke-width="12"
      style="margin: 12px 0"
    />

    <div class="chart-section">
      <h4>损失曲线</h4>
      <canvas ref="chartCanvas" width="400" height="200" class="loss-chart"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const emit = defineEmits(['status-change'])

const model = ref('deeplabv3p_resnet50')
const epochs = ref(10)
const batchSize = ref(4)
const learningRate = ref(0.001)
const preparing = ref(false)
const training = ref(false)
const samplesReady = ref(false)
const chartCanvas = ref(null)

const trainingInfo = ref({
  epoch: 0,
  loss: 0,
  miou: 0,
  status: '',
})

const lossHistory = ref([])
let pollTimer = null

async function prepareSamples() {
  preparing.value = true
  emit('status-change', '正在准备训练样本...')

  try {
    await axios.post('/api/train/prepare-samples', {
      model: model.value,
    })
    samplesReady.value = true
    emit('status-change', '训练样本准备完成')
  } catch (err) {
    emit('status-change', `样本准备失败: ${err.message}`)
  } finally {
    preparing.value = false
  }
}

async function startTraining() {
  training.value = true
  lossHistory.value = []
  trainingInfo.value = { epoch: 0, loss: 0, miou: 0, status: 'running' }
  emit('status-change', '正在启动训练...')

  try {
    const res = await axios.post('/api/train/start', {
      model: model.value,
      epochs: epochs.value,
      batch_size: batchSize.value,
      learning_rate: learningRate.value,
    })
    const taskId = res.data.task_id

    pollTimer = setInterval(async () => {
      try {
        const statusRes = await axios.get(`/api/train/status/${taskId}`)
        const data = statusRes.data

        trainingInfo.value = {
          epoch: data.epoch || 0,
          loss: data.loss || 0,
          miou: data.miou || 0,
          status: data.status || 'running',
        }

        if (data.loss !== undefined) {
          lossHistory.value.push(data.loss)
          drawChart()
        }

        emit('status-change', `训练中: epoch ${data.epoch}/${epochs.value}`)

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollTimer)
          training.value = false
          emit('status-change', data.status === 'completed' ? '训练完成' : '训练失败')
        }
      } catch {
        clearInterval(pollTimer)
        training.value = false
      }
    }, 3000)
  } catch (err) {
    training.value = false
    emit('status-change', `启动训练失败: ${err.message}`)
  }
}

function drawChart() {
  const canvas = chartCanvas.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  const w = canvas.width
  const h = canvas.height
  const data = lossHistory.value

  ctx.clearRect(0, 0, w, h)

  // Background
  ctx.fillStyle = '#fafafa'
  ctx.fillRect(0, 0, w, h)

  // Grid
  ctx.strokeStyle = '#e4e7ed'
  ctx.lineWidth = 0.5
  for (let i = 0; i <= 4; i++) {
    const y = 20 + ((h - 40) * i) / 4
    ctx.beginPath()
    ctx.moveTo(40, y)
    ctx.lineTo(w - 10, y)
    ctx.stroke()
  }

  if (data.length < 2) return

  const maxLoss = Math.max(...data, 0.1)
  const minLoss = 0
  const range = maxLoss - minLoss

  // Y-axis labels
  ctx.fillStyle = '#909399'
  ctx.font = '10px sans-serif'
  ctx.textAlign = 'right'
  for (let i = 0; i <= 4; i++) {
    const val = maxLoss - (range * i) / 4
    const y = 20 + ((h - 40) * i) / 4
    ctx.fillText(val.toFixed(3), 38, y + 3)
  }

  // Line
  ctx.beginPath()
  ctx.strokeStyle = '#1a73e8'
  ctx.lineWidth = 2

  for (let i = 0; i < data.length; i++) {
    const x = 40 + ((w - 50) * i) / (data.length - 1)
    const y = 20 + ((h - 40) * (maxLoss - data[i])) / range
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Points
  ctx.fillStyle = '#1a73e8'
  for (let i = 0; i < data.length; i++) {
    const x = 40 + ((w - 50) * i) / (data.length - 1)
    const y = 20 + ((h - 40) * (maxLoss - data[i])) / range
    ctx.beginPath()
    ctx.arc(x, y, 3, 0, Math.PI * 2)
    ctx.fill()
  }

  // X-axis label
  ctx.fillStyle = '#909399'
  ctx.textAlign = 'center'
  ctx.fillText('Epoch', w / 2, h - 2)
}

onMounted(() => {
  drawChart()
})
</script>

<style scoped>
.panel-content {
  padding: 16px;
}

h3 {
  margin: 0 0 16px;
  font-size: 16px;
  color: #303133;
}

.chart-section {
  margin-top: 16px;
}

.chart-section h4 {
  margin: 0 0 8px;
  font-size: 14px;
  color: #606266;
}

.loss-chart {
  width: 100%;
  height: 200px;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
}
</style>
