<template>
  <el-container class="shell">
    <el-header class="topbar">
      <div><div class="eyebrow">GEOAI · SEMANTIC SEGMENTATION</div><h1>U-Net 多类地物分割实验室</h1></div>
      <div class="health"><span :class="['dot', health.status]" />{{ healthText }}</div>
    </el-header>
    <el-main>
      <el-tabs v-model="active" class="workspace">
        <el-tab-pane label="数据中心" name="data">
          <section class="grid two">
            <el-card shadow="never">
              <template #header><b>离线六类数据</b></template>
              <el-radio-group v-model="profile"><el-radio-button label="quick">Quick</el-radio-button><el-radio-button label="full">Full</el-radio-button></el-radio-group>
              <p class="muted">{{ profileInfo }}</p>
              <el-button type="primary" :loading="busy" @click="generate(false)">生成数据</el-button>
              <el-button @click="generate(true)">整理真实建筑样本</el-button>
            </el-card>
            <el-card shadow="never">
              <template #header><b>类别图例</b></template>
              <div class="legend"><div v-for="item in config.classes || []" :key="item.id"><i :style="{background:`rgb(${item.color.join(',')})`}" />{{ item.id }} · {{ item.name_zh }} <small>{{ item.name }}</small></div></div>
            </el-card>
          </section>
          <el-card shadow="never" class="section"><template #header><b>数据集状态</b></template><el-table :data="datasets.datasets || []"><el-table-column prop="name" label="配置"/><el-table-column prop="path" label="路径"/><el-table-column label="状态"><template #default="scope"><el-tag :type="scope.row.ready?'success':'info'">{{scope.row.ready?'已生成':'未生成'}}</el-tag></template></el-table-column></el-table></el-card>
        </el-tab-pane>

        <el-tab-pane label="U-Net 原理" name="architecture">
          <el-alert title="核心观察：跳跃连接把高分辨率边缘直接送到解码器，使分类语义恢复为空间上精确的掩膜。" type="success" :closable="false" />
          <div class="architecture">
            <div v-for="(stage,index) in modelInfo.architecture || []" :key="stage.name" :class="['stage', stage.skip?'with-skip':'']">
              <strong>{{ stage.name }}</strong><span>{{ stage.scale }}</span><small>{{ stage.role }}</small><em v-if="stage.skip">skip</em>
              <b v-if="index < (modelInfo.architecture || []).length-1">→</b>
            </div>
          </div>
          <section class="grid two">
            <el-card shadow="never"><h3>经典 U-Net</h3><p>编码特征与同尺度解码特征拼接，细小道路和建筑边缘可以被恢复。</p></el-card>
            <el-card shadow="never"><h3>无跳连消融</h3><p>只有瓶颈语义经过连续上采样，参数规模接近，但空间细节必须从压缩表示中重建。</p></el-card>
          </section>
        </el-tab-pane>

        <el-tab-pane label="训练中心" name="train">
          <section class="grid two">
            <el-card shadow="never">
              <el-form label-position="top">
                <el-form-item label="模型"><el-select v-model="trainForm.model" style="width:100%"><el-option v-for="name in modelNames" :key="name" :label="name" :value="name"/></el-select></el-form-item>
                <el-form-item label="训练配置"><el-select v-model="trainForm.profile" style="width:100%"><el-option label="quick" value="quick"/><el-option label="full" value="full"/></el-select></el-form-item>
                <el-form-item label="数据集路径（留空自动选择）"><el-input v-model="trainForm.dataset" /></el-form-item>
                <el-checkbox v-model="trainForm.binary">真实建筑二值模式</el-checkbox>
                <div class="actions"><el-button type="primary" :loading="busy" @click="startTrain">开始训练</el-button><el-button type="danger" plain :disabled="!currentTask" @click="cancelTask">取消</el-button></div>
              </el-form>
            </el-card>
            <TaskCard :task="currentTask" />
          </section>
        </el-tab-pane>

        <el-tab-pane label="模型对比" name="compare">
          <el-card shadow="never">
            <div class="toolbar"><el-select v-model="compareProfile"><el-option label="quick" value="quick"/><el-option label="full" value="full"/></el-select><el-button type="primary" :loading="busy" @click="startCompare">顺序训练四模型</el-button></div>
            <TaskCard :task="currentTask" />
            <el-table v-if="comparison.length" :data="comparison" class="section">
              <el-table-column prop="model" label="模型"/><el-table-column label="mIoU"><template #default="s">{{fmt(s.row.metrics.miou)}}</template></el-table-column><el-table-column label="mDice"><template #default="s">{{fmt(s.row.metrics.mdice)}}</template></el-table-column><el-table-column label="边界 F1"><template #default="s">{{fmt(s.row.metrics.boundary_f1)}}</template></el-table-column><el-table-column label="参数量"><template #default="s">{{Number(s.row.metrics.parameters).toLocaleString()}}</template></el-table-column><el-table-column label="图像/秒"><template #default="s">{{fmt(s.row.metrics.images_per_second)}}</template></el-table-column>
            </el-table>
            <el-alert v-if="skipEffect" :title="`跳连收益：mIoU ${signed(skipEffect.miou_gain)}，边界 F1 ${signed(skipEffect.boundary_f1_gain)}`" type="success" :closable="false" class="section" />
          </el-card>
        </el-tab-pane>

        <el-tab-pane label="GIS 推理与成果" name="predict">
          <section class="grid two">
            <el-card shadow="never">
              <el-upload drag action="/api/uploads" :on-success="onUpload" :limit="1"><div>拖入 GeoTIFF 或点击上传</div><small>保留 CRS、仿射变换和 nodata</small></el-upload>
              <el-form label-position="top" class="section">
                <el-form-item label="输入 GeoTIFF"><el-input v-model="predictForm.input_path" /></el-form-item>
                <el-form-item label="模型检查点"><el-select v-model="predictForm.checkpoint" filterable style="width:100%"><el-option v-for="item in modelInfo.checkpoints || []" :key="item.path" :label="item.relative_path" :value="item.path"/></el-select></el-form-item>
                <div class="grid two"><el-form-item label="瓦片"><el-input-number v-model="predictForm.tile_size" :min="64" :step="64"/></el-form-item><el-form-item label="重叠"><el-input-number v-model="predictForm.overlap" :min="0" :step="16"/></el-form-item></div>
                <el-button type="primary" :loading="busy" @click="startPredict">开始推理</el-button>
              </el-form>
            </el-card>
            <el-card shadow="never"><TaskCard :task="currentTask"/><div v-if="resultLinks.length" class="downloads"><a v-for="item in resultLinks" :key="item.path" :href="`/api/download?path=${encodeURIComponent(item.path)}`">下载 {{item.label}}</a></div></el-card>
          </section>
        </el-tab-pane>
      </el-tabs>
    </el-main>
  </el-container>
</template>

<script setup>
import { computed, defineComponent, h, onMounted, reactive, ref } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const TaskCard = defineComponent({props:{task:Object},setup(props){return()=>h('div',{class:'task-card'},props.task?[h('div',{class:'task-title'},[h('b',props.task.stage||'任务'),h('span',props.task.status)]),h('div',{class:'progress'},[h('i',{style:{width:`${props.task.progress||0}%`}})]),h('p',props.task.message||''),props.task.error?h('pre',props.task.error.message):null]:[h('div',{class:'empty'},'暂无运行任务')])}})
const active=ref('data'), profile=ref('quick'), compareProfile=ref('quick'), busy=ref(false), currentTask=ref(null)
const config=reactive({}),datasets=reactive({}),modelInfo=reactive({}),health=reactive({status:'pending'})
const trainForm=reactive({model:'unet',profile:'quick',dataset:'',binary:false})
const predictForm=reactive({input_path:'',checkpoint:'',tile_size:256,overlap:64})
const comparison=ref([]),skipEffect=ref(null)
const modelNames=computed(()=> (modelInfo.models||[]).map(x=>x.name))
const healthText=computed(()=>health.status==='ok'?`后端就绪 · ${health.device}`:'后端未连接')
const profileInfo=computed(()=>{const p=config.profiles?.[profile.value];return p?`${p.samples} 张 ${p.image_size}×${p.image_size}，训练 ${p.epochs} epoch`:'加载中'})
const fmt=v=>Number(v||0).toFixed(4),signed=v=>`${v>=0?'+':''}${fmt(v)}`
const resultLinks=computed(()=>{const result=currentTask.value?.result||{};const links=[];for(const [label,path] of Object.entries(result)){if(typeof path==='string'&&/\.(tif|png|json|csv|html|zip|gpkg|geojson|pth)$/i.test(path))links.push({label,path})}if(result.vectors)for(const [label,path] of Object.entries(result.vectors))if(path)links.push({label,path});return links})
async function load(){try{Object.assign(health,(await axios.get('/api/health')).data);Object.assign(config,(await axios.get('/api/config')).data);Object.assign(datasets,(await axios.get('/api/datasets')).data);Object.assign(modelInfo,(await axios.get('/api/models')).data)}catch(e){health.status='failed'}}
async function submit(url,payload){busy.value=true;try{currentTask.value=(await axios.post(url,payload)).data;poll()}catch(e){busy.value=false;ElMessage.error(e.response?.data?.detail||e.message)}}
async function poll(){if(!currentTask.value)return;try{currentTask.value=(await axios.get(`/api/tasks/${currentTask.value.task_id}`)).data;if(['completed','failed','cancelled'].includes(currentTask.value.status)){busy.value=false;if(currentTask.value.status==='completed'){ElMessage.success('任务完成');const result=currentTask.value.result;if(result?.results){comparison.value=result.results;skipEffect.value=result.skip_effect}await load()}return}}catch(e){busy.value=false;return}setTimeout(poll,1000)}
const generate=real=>submit('/api/datasets/generate',{profile:profile.value,real_buildings:real})
const startTrain=()=>submit('/api/train',{...trainForm,dataset:trainForm.dataset||null})
const startCompare=()=>submit('/api/compare',{profile:compareProfile.value,models:['unet','no_skip','unetpp','deeplabv3plus']})
const startPredict=()=>submit('/api/predict',predictForm)
async function cancelTask(){if(currentTask.value)currentTask.value=(await axios.post(`/api/tasks/${currentTask.value.task_id}/cancel`)).data}
function onUpload(response){predictForm.input_path=response.path;ElMessage.success('GeoTIFF 已上传')}
onMounted(load)
</script>
