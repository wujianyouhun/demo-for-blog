<template>
  <div class="portal">
    <div class="portal-actions"><b>统一示例门户</b><el-button size="small" @click="load(true)">探测运行状态</el-button></div>
    <el-alert title="从数据下载到成果质检；各应用端口独立，U-Net 是完整教学与工程入口。" type="info" :closable="false"/>
    <div class="grid">
      <el-card v-for="item in projects" :key="item.id" shadow="hover" :class="{unet:item.id==='unet-segmentation'}">
        <div class="card-head"><strong>{{ item.name }}</strong><el-tag size="small" :type="item.runtime==='running'?'success':'info'">{{ item.runtime || item.status }}</el-tag></div>
        <p>{{ item.category }} · {{ item.path }}</p>
        <div><el-link :href="item.frontend" target="_blank" type="primary">打开 Web</el-link><el-link :href="item.backend + '/docs'" target="_blank">API</el-link></div>
      </el-card>
    </div>
  </div>
</template>
<script setup>
import {ref,onMounted} from 'vue'
import axios from 'axios'
const projects=ref([])
async function load(probe=false){projects.value=(await axios.get('/api/projects',{params:{probe}})).data.projects}
onMounted(()=>load(false))
</script>
<style scoped>
.portal{padding:16px}.portal-actions,.card-head{display:flex;align-items:center;justify-content:space-between}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}.grid p{color:#667085;font-size:12px}.grid a{margin-right:16px}.unet{border:2px solid #2a9d8f;background:#f1fffb}
</style>
