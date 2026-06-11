import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // SAM 推理可能较慢
})

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
export function predictByText(sessionId, text, boxThreshold = 0.25, textThreshold = 0.25) {
  return api.post('/predict/text', {
    session_id: sessionId,
    text,
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

/**
 * 清除 Mask
 */
export function clearSession(sessionId) {
  return api.delete('/session/clear', { params: { session_id: sessionId } })
}
