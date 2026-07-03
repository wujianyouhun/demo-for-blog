import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // SAM 推理可能较慢
})

export function getConfig() {
  return api.get('/config')
}

/**
 * 加载影像
 */
export function loadImage(imagePath, modelType = 'vit_l', samVersion = 'sam1') {
  return api.post('/image/load', {
    image_path: imagePath,
    model_type: modelType,
    sam_version: samVersion,
  })
}

/**
 * 获取影像元数据（含 EPSG:3857 extent）
 */
export function getImageInfo(sessionId) {
  return api.get('/image/info', { params: { session_id: sessionId } })
}

/**
 * 点提示分割 — 返回 blob (PNG mask)
 */
export function predictByPoint(sessionId, points, labels) {
  return api.post('/predict/point', {
    session_id: sessionId,
    points,
    labels,
  }, { responseType: 'blob' })
}

/**
 * 框提示分割 — 返回 blob (PNG mask)
 */
export function predictByBox(sessionId, box) {
  return api.post('/predict/box', {
    session_id: sessionId,
    box,
  }, { responseType: 'blob' })
}

/**
 * 文本提示分割
 */
export function predictByText(sessionId, text, boxThreshold = 0.25, textThreshold = 0.25, clickPoint = null) {
  return api.post('/predict/text', {
    session_id: sessionId,
    text,
    lon: clickPoint?.lon ?? null,
    lat: clickPoint?.lat ?? null,
    box_threshold: boxThreshold,
    text_threshold: textThreshold,
  }, { responseType: 'blob' })
}

/**
 * 后处理 Mask
 */
export function postprocessMask(sessionId, params = {}) {
  return api.post('/postprocess', {
    session_id: sessionId,
    ...params,
  }, { responseType: 'blob' })
}

/**
 * 矢量化导出
 */
export function exportVectors(sessionId, minArea = 50, outputFormat = 'geojson') {
  return api.post('/export/vectorize', {
    session_id: sessionId,
    min_area: minArea,
    output_format: outputFormat,
  })
}

/**
 * 下载导出文件
 */
export function downloadExport(sessionId) {
  return api.get('/export/download', {
    params: { session_id: sessionId },
    responseType: 'blob',
  })
}

/**
 * 会话状态
 */
export function getSessionStatus(sessionId) {
  return api.get('/session/status', { params: { session_id: sessionId } })
}

export function getSessionProgress(sessionId) {
  return api.get('/session/progress', { params: { session_id: sessionId } })
}

export function getBackendLogs(limit = 80) {
  return api.get('/logs', { params: { limit } })
}

/**
 * 清除 Mask
 */
export function clearSession(sessionId) {
  return api.delete('/session/clear', { params: { session_id: sessionId } })
}

/**
 * 启动整幅影像瓦片化处理任务
 */
export function startFullProcess(params) {
  return api.post('/process/full', params)
}

/**
 * 查询整幅影像处理任务状态
 */
export function getFullProcessStatus(taskId) {
  return api.get('/process/status', { params: { task_id: taskId } })
}

/**
 * 下载整幅影像处理结果
 */
export function downloadFullProcessResult(taskId) {
  return api.get('/process/download', {
    params: { task_id: taskId },
    responseType: 'blob',
  })
}
