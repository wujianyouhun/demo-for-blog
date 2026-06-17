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

/** 执行正则化流水线 */
export function runPipeline(config) {
  return api.post('/run', { config }).then(r => r.data)
}

/** 获取单步结果 */
export function getStep(key) {
  return api.get(`/step/${key}`).then(r => r.data)
}

/** 获取统计对比 */
export function getCompareStats() {
  return api.get('/compare').then(r => r.data)
}

/** 导出结果 */
export function exportResult(fmt) {
  return api.get(`/export/${fmt}`, { responseType: 'blob' }).then(r => {
    const blob = new Blob([r.data])
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const ext = { geojson: '.geojson', gpkg: '.gpkg', shp: '.zip' }[fmt] || '.geojson'
    a.download = `regularized${ext}`
    a.click()
    URL.revokeObjectURL(url)
  })
}
