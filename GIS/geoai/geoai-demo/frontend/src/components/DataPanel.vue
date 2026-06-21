<template>
  <div class="panel-content">
    <h3>数据下载与管理</h3>

    <el-form label-width="90px" size="default">
      <el-form-item label="选择区域">
        <el-select v-model="region" placeholder="选择预设区域" style="width: 100%">
          <el-option label="北京 (Beijing)" value="beijing" />
          <el-option label="上海 (Shanghai)" value="shanghai" />
          <el-option label="深圳 (Shenzhen)" value="shenzhen" />
          <el-option label="成都 (Chengdu)" value="chengdu" />
          <el-option label="武汉 (Wuhan)" value="wuhan" />
          <el-option label="自定义 (Custom)" value="custom" />
        </el-select>
      </el-form-item>

      <el-form-item v-if="region === 'custom'" label="自定义范围">
        <el-row :gutter="8">
          <el-col :span="12">
            <el-input v-model.number="bbox.minLon" placeholder="Min Lon" size="small" />
          </el-col>
          <el-col :span="12">
            <el-input v-model.number="bbox.minLat" placeholder="Min Lat" size="small" />
          </el-col>
        </el-row>
        <el-row :gutter="8" style="margin-top: 4px">
          <el-col :span="12">
            <el-input v-model.number="bbox.maxLon" placeholder="Max Lon" size="small" />
          </el-col>
          <el-col :span="12">
            <el-input v-model.number="bbox.maxLat" placeholder="Max Lat" size="small" />
          </el-col>
        </el-row>
      </el-form-item>

      <el-form-item label="日期范围">
        <el-date-picker
          v-model="dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          format="YYYY-MM-DD"
          value-format="YYYY-MM-DD"
          style="width: 100%"
        />
      </el-form-item>

      <el-form-item label="云量上限">
        <el-slider v-model="cloudCover" :min="0" :max="100" :step="5" show-input />
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="downloading" @click="handleDownload">
          下载数据
        </el-button>
        <el-button @click="refreshFiles">刷新列表</el-button>
      </el-form-item>
    </el-form>

    <el-progress
      v-if="downloading"
      :percentage="progress"
      :status="progress >= 100 ? 'success' : ''"
      style="margin-bottom: 16px"
    />

    <el-divider>已下载文件</el-divider>

    <el-table :data="files" size="small" max-height="200" empty-text="暂无文件">
      <el-table-column prop="name" label="文件名" />
      <el-table-column prop="size" label="大小" width="80" />
      <el-table-column prop="date" label="日期" width="110" />
    </el-table>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const emit = defineEmits(['status-change'])

const region = ref('beijing')
const bbox = ref({ minLon: 116.2, minLat: 39.7, maxLon: 116.6, maxLat: 40.1 })
const dateRange = ref(['2023-06-01', '2023-08-31'])
const cloudCover = ref(20)
const downloading = ref(false)
const progress = ref(0)
const files = ref([])

let pollTimer = null

async function handleDownload() {
  downloading.value = true
  progress.value = 0
  emit('status-change', '正在请求下载数据...')

  try {
    const payload = {
      region: region.value,
      bbox: region.value === 'custom' ? bbox.value : null,
      date_start: dateRange.value?.[0] || '2023-06-01',
      date_end: dateRange.value?.[1] || '2023-08-31',
      cloud_cover_max: cloudCover.value,
    }

    const res = await axios.post('/api/data/download', payload)
    const taskId = res.data.task_id

    pollProgress(taskId)
  } catch (err) {
    downloading.value = false
    emit('status-change', `下载请求失败: ${err.message}`)
  }
}

function pollProgress(taskId) {
  pollTimer = setInterval(async () => {
    try {
      const res = await axios.get(`/api/data/download/status/${taskId}`)
      progress.value = Math.round((res.data.progress || 0) * 100)
      emit('status-change', `下载中: ${progress.value}%`)

      if (res.data.status === 'completed' || progress.value >= 100) {
        clearInterval(pollTimer)
        downloading.value = false
        progress.value = 100
        emit('status-change', '数据下载完成')
        refreshFiles()
      } else if (res.data.status === 'failed') {
        clearInterval(pollTimer)
        downloading.value = false
        emit('status-change', '下载失败')
      }
    } catch {
      clearInterval(pollTimer)
      downloading.value = false
    }
  }, 2000)
}

async function refreshFiles() {
  try {
    const res = await axios.get('/api/data/files')
    files.value = res.data.files || []
  } catch {
    files.value = []
  }
}

onMounted(() => {
  refreshFiles()
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
