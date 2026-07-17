<template>
  <main :style="{'--accent':config.accent||'#168aad'}">
    <header><div><small>{{ config.category || 'GEOAI WEB EXAMPLE' }}</small><h1>{{ config.title }}</h1><p>{{ config.description }}</p></div><span :class="['health',health.status]">{{health.status==='ok'?'服务就绪':'连接中'}}</span></header>
    <section class="layout">
      <el-card shadow="never" class="operations">
        <el-tabs v-model="active">
          <el-tab-pane v-for="operation in config.operations" :key="operation.id" :label="operation.title" :name="operation.id">
            <p class="description">{{operation.description}}</p>
            <el-form label-position="top">
              <template v-for="field in operation.fields||[]" :key="field.key">
                <el-form-item v-if="field.type==='select'" :label="field.label"><el-select v-model="forms[operation.id][field.key]" style="width:100%"><el-option v-for="option in field.options" :key="option.value??option" :label="option.label??option" :value="option.value??option"/></el-select></el-form-item>
                <el-form-item v-else-if="field.type==='number'" :label="field.label"><el-input-number v-model="forms[operation.id][field.key]" :min="field.min" :max="field.max" :step="field.step||1"/></el-form-item>
                <el-form-item v-else-if="field.type==='checkbox'"><el-checkbox v-model="forms[operation.id][field.key]">{{field.label}}</el-checkbox></el-form-item>
                <el-form-item v-else-if="field.type==='upload'" :label="field.label"><el-upload drag :action="field.endpoint||'/api/uploads'" :on-success="response=>onUpload(operation.id,field.key,response)"><div>拖入文件或点击上传</div></el-upload><el-input v-model="forms[operation.id][field.key]" class="upload-path"/></el-form-item>
                <el-form-item v-else :label="field.label"><el-input v-model="forms[operation.id][field.key]" :type="field.type==='textarea'?'textarea':'text'" :placeholder="field.placeholder"/></el-form-item>
              </template>
              <el-button type="primary" :loading="busy" @click="run(operation)">{{operation.button||'执行'}}</el-button>
              <el-button v-if="task" type="danger" plain @click="cancel">取消任务</el-button>
            </el-form>
          </el-tab-pane>
        </el-tabs>
      </el-card>
      <aside>
        <el-card shadow="never"><template #header><b>任务状态</b></template><div v-if="task"><div class="task-head"><b>{{task.stage}}</b><el-tag>{{task.status}}</el-tag></div><el-progress :percentage="task.progress||0"/><p>{{task.message}}</p><pre v-if="task.error">{{task.error.message}}</pre></div><el-empty v-else description="尚未运行任务" :image-size="70"/></el-card>
        <el-card shadow="never" class="result-card"><template #header><b>结果与文件</b></template><div v-if="result"><pre>{{pretty(result)}}</pre><div class="downloads"><a v-for="file in resultFiles" :key="file.path" :href="`/api/download?path=${encodeURIComponent(file.path)}`">下载 {{file.label}}</a></div></div><el-empty v-else description="暂无结果" :image-size="70"/></el-card>
      </aside>
    </section>
    <el-card shadow="never" class="files"><template #header><div class="task-head"><b>项目文件</b><el-button size="small" @click="loadFiles">刷新</el-button></div></template><el-table :data="files"><el-table-column prop="name" label="文件"/><el-table-column prop="relative_path" label="相对路径"/><el-table-column prop="suffix" label="格式" width="90"/><el-table-column label="大小" width="120"><template #default="s">{{size(s.row.size)}}</template></el-table-column></el-table></el-card>
  </main>
</template>
<script setup>
import{computed,onMounted,reactive,ref}from'vue';import axios from'axios';import{ElMessage}from'element-plus';
const props=defineProps({config:{type:Object,required:true}});const config=props.config,health=reactive({status:'pending'}),active=ref(config.operations[0]?.id),forms=reactive({}),task=ref(null),result=ref(null),files=ref([]),busy=ref(false)
for(const operation of config.operations){forms[operation.id]={};for(const field of operation.fields||[])forms[operation.id][field.key]=field.default??(field.type==='checkbox'?false:field.type==='number'?0:'')}
const resultFiles=computed(()=>{const out=[];function walk(value,key='result'){if(typeof value==='string'&&/\.(tif|tiff|png|jpg|geojson|gpkg|shp|zip|json|csv|html|pth|pt)$/i.test(value))out.push({label:key,path:value});else if(value&&typeof value==='object')for(const[k,v]of Object.entries(value))walk(v,k)}walk(result.value);return out})
async function load(){try{Object.assign(health,(await axios.get('/api/health')).data);await loadFiles()}catch{health.status='failed'}}async function loadFiles(){try{const payload=(await axios.get('/api/files')).data;files.value=payload.files||payload}catch{files.value=[]}}
async function run(operation){busy.value=true;result.value=null;try{const response=await axios.post(operation.endpoint,forms[operation.id]);if(response.data.task_id){task.value=response.data;poll()}else{result.value=response.data;busy.value=false;ElMessage.success('完成')}}catch(error){busy.value=false;ElMessage.error(error.response?.data?.detail||error.message)}}
async function poll(){try{task.value=(await axios.get(`/api/tasks/${task.value.task_id}`)).data;if(['completed','failed','cancelled'].includes(task.value.status)){busy.value=false;result.value=task.value.result;if(task.value.status==='completed'){ElMessage.success('任务完成');loadFiles()}return}}catch{busy.value=false;return}setTimeout(poll,1000)}async function cancel(){if(task.value)task.value=(await axios.post(`/api/tasks/${task.value.task_id}/cancel`)).data}
function onUpload(operation,key,response){forms[operation][key]=response.path;ElMessage.success('文件已上传')}const pretty=v=>JSON.stringify(v,null,2);const size=v=>v>1048576?`${(v/1048576).toFixed(1)} MB`:v>1024?`${(v/1024).toFixed(1)} KB`:`${v||0} B`;onMounted(load)
</script>
