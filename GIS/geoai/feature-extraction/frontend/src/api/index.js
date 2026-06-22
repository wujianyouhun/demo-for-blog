import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

/** 获取 TIF 元数据 */
export function getTifInfo() {
  return api.get('/tif-info').then(r => r.data)
}

/** 预生成瓦片 */
export function generateTiles(zoomLevels) {
  return api.post('/generate-tiles', null, {
    params: zoomLevels ? { zoom_levels: zoomLevels } : {}
  }).then(r => r.data)
}

/** 执行要素提取 */
export function extractFeatures(config) {
  return api.post('/extract', config, { timeout: 600000 }).then(r => r.data)
}

/** 获取提取结果 */
export function getResult(target) {
  return api.get('/result', { params: target ? { target } : {} }).then(r => r.data)
}

/** 获取统计信息 */
export function getStats() {
  return api.get('/stats').then(r => r.data)
}

/** 导出结果 */
export function exportResult(fmt, target) {
  return api.get(`/export/${fmt}`, {
    responseType: 'blob',
    params: target ? { target } : {},
  }).then(r => {
    const blob = new Blob([r.data])
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const ext = { geojson: '.geojson', gpkg: '.gpkg', shp: '.zip' }[fmt] || '.geojson'
    a.download = `extraction${ext}`
    a.click()
    URL.revokeObjectURL(url)
  })
}
