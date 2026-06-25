/**
 * GeoAI 遥感图像分类系统 — 前端逻辑
 * API Base: http://localhost:8000
 */

const API = 'http://localhost:8000';

// ─── 类别 Emoji 映射 ────────────────────────────────────────────────
const CLASS_EMOJI = {
  AnnualCrop:             '🌾',
  Forest:                 '🌲',
  HerbaceousVegetation:   '🌿',
  Highway:                '🛣️',
  Industrial:             '🏭',
  Pasture:                '🐄',
  PermanentCrop:          '🍇',
  Residential:            '🏘️',
  River:                  '🌊',
  SeaLake:                '🏖️',
};

// ─── DOM 引用 ────────────────────────────────────────────────────────
const dropzone     = document.getElementById('dropzone');
const fileInput    = document.getElementById('fileInput');
const previewWrap  = document.getElementById('previewWrap');
const previewImg   = document.getElementById('previewImg');
const clearBtn     = document.getElementById('clearBtn');
const predictBtn   = document.getElementById('predictBtn');
const progressBar  = document.getElementById('progressBar');
const errorBox     = document.getElementById('errorBox');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const mainResult   = document.getElementById('mainResult');
const resultEmoji  = document.getElementById('resultEmoji');
const resultClass  = document.getElementById('resultClass');
const resultClassZh = document.getElementById('resultClassZh');
const resultConf   = document.getElementById('resultConf');
const resultDesc   = document.getElementById('resultDesc');
const inferTime    = document.getElementById('inferTime');
const modelInfo    = document.getElementById('modelInfo');
const top5Wrap     = document.getElementById('top5Wrap');
const top5List     = document.getElementById('top5List');
const statusDot    = document.getElementById('statusDot');
const statusText   = document.getElementById('statusText');
const classesList  = document.getElementById('classesList');
const batchInput   = document.getElementById('batchInput');
const batchBtn     = document.getElementById('batchBtn');
const batchStatus  = document.getElementById('batchStatus');
const batchResults = document.getElementById('batchResults');

let currentFile = null;
let batchFiles  = [];
let modelName   = 'ResNet50';

// ─── 服务健康检查 ────────────────────────────────────────────────────
async function checkHealth() {
  statusDot.className  = 'status-dot loading';
  statusText.textContent = '连接中...';
  try {
    const res  = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    if (data.loaded) {
      statusDot.className    = 'status-dot online';
      statusText.textContent = `在线 · ${data.model_name}`;
      modelName = data.model_name;
      modelInfo.textContent  = `🧠 ${data.model_name} · ${data.num_classes} 类`;
    } else {
      statusDot.className    = 'status-dot offline';
      statusText.textContent = '模型未加载';
    }
  } catch {
    statusDot.className    = 'status-dot offline';
    statusText.textContent = '服务离线';
  }
}

// ─── 加载类别列表 ────────────────────────────────────────────────────
async function loadClasses() {
  try {
    const res  = await fetch(`${API}/classes`);
    const data = await res.json();
    classesList.innerHTML = data.classes.map(c => `
      <div class="class-chip">
        <span class="class-emoji">${CLASS_EMOJI[c.name] || '🗺️'}</span>
        <div>
          <div>${c.name}</div>
          <div class="class-name-zh">${c.name_zh}</div>
        </div>
      </div>
    `).join('');
  } catch {
    // 使用本地备用
    const fallback = [
      ['AnnualCrop','农田'],['Forest','森林'],['HerbaceousVegetation','草本植被'],
      ['Highway','公路'],['Industrial','工业区'],['Pasture','牧场'],
      ['PermanentCrop','永久作物'],['Residential','居民区'],['River','河流'],['SeaLake','海湖'],
    ];
    classesList.innerHTML = fallback.map(([name,zh]) => `
      <div class="class-chip">
        <span class="class-emoji">${CLASS_EMOJI[name] || '🗺️'}</span>
        <div><div>${name}</div><div class="class-name-zh">${zh}</div></div>
      </div>
    `).join('');
  }
}

// ─── 文件处理 ────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/') &&
      !file.name.endsWith('.tif') && !file.name.endsWith('.tiff')) {
    showError('请选择有效的图像文件 (JPG/PNG/TIFF)');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('文件过大，最大支持 10 MB');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.classList.remove('hidden');
    dropzone.classList.add('hidden');
    predictBtn.disabled = false;
    hideError();
    clearResult();
  };
  reader.readAsDataURL(file);
}

function clearAll() {
  currentFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.classList.add('hidden');
  dropzone.classList.remove('hidden');
  predictBtn.disabled = true;
  clearResult();
  hideError();
}

function clearResult() {
  mainResult.classList.add('hidden');
  top5Wrap.classList.add('hidden');
  resultPlaceholder.classList.remove('hidden');
}

// ─── 拖拽上传 ────────────────────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', ()=> dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
clearBtn.addEventListener('click', clearAll);

// ─── 单张预测 ────────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  setLoading(true);
  hideError();
  try {
    const fd = new FormData();
    fd.append('file', currentFile);
    const res = await fetch(`${API}/predict`, {
      method: 'POST', body: fd,
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    showError(`预测失败: ${e.message}`);
    clearResult();
  } finally {
    setLoading(false);
  }
});

function renderResult(data) {
  resultPlaceholder.classList.add('hidden');
  mainResult.classList.remove('hidden');
  top5Wrap.classList.remove('hidden');

  const confPct  = (data.confidence * 100).toFixed(1) + '%';
  resultEmoji.textContent    = CLASS_EMOJI[data.class_name] || '🗺️';
  resultClass.textContent    = data.class_name;
  resultClassZh.textContent  = data.class_name_zh || '';
  resultConf.textContent     = confPct;
  resultConf.style.color     = data.confidence > 0.8 ? 'var(--success)'
                             : data.confidence > 0.5 ? 'var(--warn)' : 'var(--danger)';
  resultDesc.textContent     = data.description || '';
  inferTime.textContent      = `⏱ ${data.infer_time_ms} ms`;
  modelInfo.textContent      = `🧠 ${modelName}`;

  // Top-5 条形图
  const maxProb = data.top5[0].prob;
  top5List.innerHTML = data.top5.map((item, i) => `
    <div class="top5-item">
      <div class="top5-label">${CLASS_EMOJI[item.class] || ''} ${item.class_zh || item.class}</div>
      <div class="top5-bar-wrap">
        <div class="top5-bar${i === 0 ? ' first' : ''}"
             style="width:${(item.prob / maxProb * 100).toFixed(1)}%"></div>
      </div>
      <div class="top5-prob">${(item.prob * 100).toFixed(1)}%</div>
    </div>
  `).join('');
}

// ─── 批量预测 ────────────────────────────────────────────────────────
batchInput.addEventListener('change', () => {
  batchFiles = Array.from(batchInput.files).slice(0, 16);
  batchBtn.disabled = batchFiles.length === 0;
  batchStatus.textContent = batchFiles.length > 0
    ? `已选择 ${batchFiles.length} 张图像` : '';
  batchResults.innerHTML = '';
});

batchBtn.addEventListener('click', async () => {
  if (batchFiles.length === 0) return;
  batchBtn.disabled  = true;
  batchResults.innerHTML = '';
  batchStatus.textContent = `正在分类 ${batchFiles.length} 张图像...`;

  const fd = new FormData();
  batchFiles.forEach(f => fd.append('files', f));

  try {
    const res = await fetch(`${API}/predict/batch`, {
      method: 'POST', body: fd,
      signal: AbortSignal.timeout(60000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    batchStatus.textContent = `✅ 完成 ${data.count} 张`;
    batchResults.innerHTML = data.results.map(item => item.success ? `
      <div class="batch-item ok">
        <div class="batch-filename">${item.filename}</div>
        <div class="batch-cls">${CLASS_EMOJI[item.result.class_name] || ''} ${item.result.class_name}</div>
        <div class="batch-conf">${item.result.class_name_zh} · ${(item.result.confidence*100).toFixed(1)}%</div>
      </div>
    ` : `
      <div class="batch-item err">
        <div class="batch-filename">${item.filename}</div>
        <div style="color:var(--danger);font-size:.85rem">❌ ${item.error}</div>
      </div>
    `).join('');
  } catch (e) {
    batchStatus.textContent = `❌ 批量预测失败: ${e.message}`;
  } finally {
    batchBtn.disabled = false;
  }
});

// ─── 工具函数 ────────────────────────────────────────────────────────
function setLoading(on) {
  predictBtn.disabled = on;
  predictBtn.classList.toggle('loading', on);
  predictBtn.innerHTML = on
    ? '<span class="btn-icon">⏳</span>分类中...'
    : '<span class="btn-icon">🔍</span>开始分类';
  progressBar.classList.toggle('hidden', !on);
}
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}
function hideError() {
  errorBox.classList.add('hidden');
}

// ─── 初始化 ──────────────────────────────────────────────────────────
(async () => {
  await Promise.all([checkHealth(), loadClasses()]);
  setInterval(checkHealth, 30000);
})();
