import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

/** 获取演示数据列表 */
export function listDemos() {
  return api.get('/demo-list').then(r => r.data)
}

/** 加载指定演示数据 */
export function loadDemo(name) {
  return api.get(`/demo/${name}`).then(r => r.data)
}

/** 上传 GeoJSON 文件 */
export function uploadGeoJSON(file) {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

/** 执行质量检查 */
export function runCheck(params) {
  return api.post('/check', params).then(r => r.data)
}

/** 执行一键修复 */
export function runRepair(config) {
  return api.post('/repair', config).then(r => r.data)
}

/** 获取修复步骤结果 */
export function getRepairStep(key) {
  return api.get(`/repair-step/${key}`).then(r => r.data)
}

/** 获取检查报告 */
export function getReport() {
  return api.get('/report').then(r => r.data)
}

/** 导出修复后数据 */
export function exportResult(fmt) {
  return api.get(`/export/${fmt}`, { responseType: 'blob' }).then(r => {
    const blob = new Blob([r.data])
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const ext = { geojson: '.geojson', gpkg: '.gpkg', shp: '.zip' }[fmt] || '.geojson'
    a.download = `repaired${ext}`
    a.click()
    URL.revokeObjectURL(url)
  })
}
