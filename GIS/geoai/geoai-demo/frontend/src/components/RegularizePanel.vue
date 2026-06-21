<template>
  <div class="panel-content">
    <h3>要素正则化</h3>

    <el-form label-width="100px" size="default">
      <el-form-item label="简化容差">
        <el-slider v-model="simplifyTolerance" :min="0.1" :max="10" :step="0.1" show-input />
      </el-form-item>

      <el-form-item label="平滑迭代">
        <el-input-number v-model="smoothIterations" :min="0" :max="10" :step="1" style="width: 100%" />
      </el-form-item>

      <el-form-item label="最小面积(m²)">
        <el-input-number v-model="minArea" :min="0" :max="10000" :step="10" style="width: 100%" />
      </el-form-item>

      <el-form-item label="正交化">
        <el-switch v-model="orthogonalize" active-text="启用" inactive-text="关闭" />
      </el-form-item>

      <el-form-item label="正交化角度">
        <el-input-number
          v-model="orthoAngle"
          :min="1"
          :max="45"
          :step="1"
          :disabled="!orthogonalize"
          style="width: 100%"
        />
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="running" @click="runRegularize">
          运行正则化
        </el-button>
      </el-form-item>
    </el-form>

    <el-divider>统计对比</el-divider>

    <el-row :gutter="12">
      <el-col :span="12">
        <el-card shadow="hover" class="stat-card">
          <template #header>
            <span class="stat-header">处理前</span>
          </template>
          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="图斑数">{{ statsBefore.count }}</el-descriptions-item>
            <el-descriptions-item label="总面积">
              {{ statsBefore.totalArea.toFixed(2) }} m²
            </el-descriptions-item>
            <el-descriptions-item label="平均面积">
              {{ statsBefore.avgArea.toFixed(2) }} m²
            </el-descriptions-item>
            <el-descriptions-item label="顶点数">{{ statsBefore.vertices }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="hover" class="stat-card">
          <template #header>
            <span class="stat-header">处理后</span>
          </template>
          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="图斑数">{{ statsAfter.count }}</el-descriptions-item>
            <el-descriptions-item label="总面积">
              {{ statsAfter.totalArea.toFixed(2) }} m²
            </el-descriptions-item>
            <el-descriptions-item label="平均面积">
              {{ statsAfter.avgArea.toFixed(2) }} m²
            </el-descriptions-item>
            <el-descriptions-item label="顶点数">{{ statsAfter.vertices }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <div v-if="running" style="margin-top: 16px">
      <el-progress :percentage="regProgress" :stroke-width="12" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const emit = defineEmits(['show-result', 'status-change'])

const simplifyTolerance = ref(1.0)
const smoothIterations = ref(2)
const minArea = ref(50)
const orthogonalize = ref(true)
const orthoAngle = ref(10)
const running = ref(false)
const regProgress = ref(0)

const statsBefore = ref({ count: 0, totalArea: 0, avgArea: 0, vertices: 0 })
const statsAfter = ref({ count: 0, totalArea: 0, avgArea: 0, vertices: 0 })

async function runRegularize() {
  running.value = true
  regProgress.value = 0
  emit('status-change', '正在执行正则化...')

  try {
    const res = await axios.post('/api/extract/regularize', {
      simplify_tolerance: simplifyTolerance.value,
      smooth_iterations: smoothIterations.value,
      min_area: minArea.value,
      orthogonalize: orthogonalize.value,
      ortho_angle: orthoAngle.value,
    })

    const taskId = res.data.task_id

    const pollTimer = setInterval(async () => {
      try {
        const statusRes = await axios.get(`/api/extract/regularize/status/${taskId}`)
        const data = statusRes.data

        regProgress.value = Math.round((data.progress || 0) * 100)
        emit('status-change', `正则化中: ${regProgress.value}%`)

        if (data.status === 'completed') {
          clearInterval(pollTimer)
          running.value = false
          regProgress.value = 100

          if (data.stats_before) {
            statsBefore.value = {
              count: data.stats_before.count || 0,
              totalArea: data.stats_before.total_area || 0,
              avgArea: data.stats_before.avg_area || 0,
              vertices: data.stats_before.vertices || 0,
            }
          }

          if (data.stats_after) {
            statsAfter.value = {
              count: data.stats_after.count || 0,
              totalArea: data.stats_after.total_area || 0,
              avgArea: data.stats_after.avg_area || 0,
              vertices: data.stats_after.vertices || 0,
            }
          }

          if (data.geojson) {
            emit('show-result', data.geojson)
          }

          emit('status-change', '正则化完成')
        } else if (data.status === 'failed') {
          clearInterval(pollTimer)
          running.value = false
          emit('status-change', '正则化失败')
        }
      } catch {
        clearInterval(pollTimer)
        running.value = false
      }
    }, 2000)
  } catch (err) {
    running.value = false
    emit('status-change', `正则化请求失败: ${err.message}`)
  }
}
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

.stat-card {
  height: 100%;
}

.stat-header {
  font-weight: 600;
  font-size: 14px;
}
</style>
