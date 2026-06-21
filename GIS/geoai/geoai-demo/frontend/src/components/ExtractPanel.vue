<template>
  <div class="panel-content">
    <h3>要素提取 (语义分割推理)</h3>

    <el-form label-width="90px" size="default">
      <el-form-item label="输入影像">
        <el-select v-model="inputImage" placeholder="选择影像文件" style="width: 100%">
          <el-option
            v-for="f in imageFiles"
            :key="f.name"
            :label="f.name"
            :value="f.path"
          />
        </el-select>
      </el-form-item>

      <el-form-item label="推理模型">
        <el-select v-model="modelFile" placeholder="选择模型" style="width: 100%">
          <el-option
            v-for="m in modelFiles"
            :key="m.name"
            :label="m.name"
            :value="m.path"
          />
        </el-select>
      </el-form-item>

      <el-form-item label="切片尺寸">
        <el-input-number v-model="tileSize" :min="128" :max="1024" :step="128" style="width: 100%" />
      </el-form-item>

      <el-form-item label="重叠区域">
        <el-input-number v-model="overlap" :min="0" :max="256" :step="16" style="width: 100%" />
      </el-form-item>

      <el-form-item label="置信阈值">
        <el-slider v-model="threshold" :min="0" :max="1" :step="0.05" show-input />
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="extracting" @click="startExtract">
          开始提取
        </el-button>
        <el-button @click="refreshLists">刷新列表</el-button>
      </el-form-item>
    </el-form>

    <el-divider>提取进度</el-divider>

    <el-progress
      v-if="extracting"
      :percentage="extractProgress"
      :stroke-width="12"
      style="margin-bottom: 12px"
    />

    <el-descriptions :column="1" border size="small" v-if="resultInfo">
      <el-descriptions-item label="总图斑数">{{ resultInfo.total_features }}</el-descriptions-item>
      <el-descriptions-item label="处理耗时">{{ resultInfo.elapsed }}s</el-descriptions-item>
      <el-descriptions-item label="输出文件">{{ resultInfo.output_file }}</el-descriptions-item>
    </el-descriptions>

    <el-divider>类别统计</el-divider>

    <el-table :data="classStats" size="small" empty-text="暂无数据" max-height="180">
      <el-table-column prop="name" label="类别" />
      <el-table-column prop="count" label="图斑数" width="80" />
      <el-table-column prop="area" label="面积(m²)" width="110">
        <template #default="{ row }">
          {{ row.area.toFixed(2) }}
        </template>
      </el-table-column>
      <el-table-column label="占比" width="80">
        <template #default="{ row }">
          {{ (row.ratio * 100).toFixed(1) }}%
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const emit = defineEmits(['show-result', 'status-change'])

const inputImage = ref('')
const modelFile = ref('')
const tileSize = ref(256)
const overlap = ref(32)
const threshold = ref(0.5)
const extracting = ref(false)
const extractProgress = ref(0)
const resultInfo = ref(null)
const classStats = ref([])
const imageFiles = ref([])
const modelFiles = ref([])

let pollTimer = null

async function refreshLists() {
  try {
    const [imgRes, modelRes] = await Promise.all([
      axios.get('/api/data/files?type=image'),
      axios.get('/api/train/models'),
    ])
    imageFiles.value = imgRes.data.files || []
    modelFiles.value = modelRes.data.models || []
  } catch {
    imageFiles.value = []
    modelFiles.value = []
  }
}

async function startExtract() {
  if (!inputImage.value) {
    emit('status-change', '请先选择输入影像')
    return
  }
  if (!modelFile.value) {
    emit('status-change', '请先选择模型')
    return
  }

  extracting.value = true
  extractProgress.value = 0
  resultInfo.value = null
  classStats.value = []
  emit('status-change', '正在执行推理...')

  try {
    const res = await axios.post('/api/extract/predict', {
      input_image: inputImage.value,
      model_path: modelFile.value,
      tile_size: tileSize.value,
      overlap: overlap.value,
      threshold: threshold.value,
    })

    const taskId = res.data.task_id
    pollTimer = setInterval(async () => {
      try {
        const statusRes = await axios.get(`/api/extract/status/${taskId}`)
        const data = statusRes.data

        extractProgress.value = Math.round((data.progress || 0) * 100)
        emit('status-change', `推理中: ${extractProgress.value}%`)

        if (data.status === 'completed') {
          clearInterval(pollTimer)
          extracting.value = false
          extractProgress.value = 100

          resultInfo.value = {
            total_features: data.total_features || 0,
            elapsed: (data.elapsed || 0).toFixed(1),
            output_file: data.output_file || '',
          }

          classStats.value = data.class_stats || []

          if (data.geojson) {
            emit('show-result', data.geojson)
          }

          emit('status-change', '要素提取完成')
        } else if (data.status === 'failed') {
          clearInterval(pollTimer)
          extracting.value = false
          emit('status-change', '要素提取失败')
        }
      } catch {
        clearInterval(pollTimer)
        extracting.value = false
      }
    }, 2000)
  } catch (err) {
    extracting.value = false
    emit('status-change', `推理请求失败: ${err.message}`)
  }
}

onMounted(() => {
  refreshLists()
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
</style>
