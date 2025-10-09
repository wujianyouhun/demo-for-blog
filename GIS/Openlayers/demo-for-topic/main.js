/**
 * 农田用水专题图应用
 *
 * 功能说明：
 * 1. 支持天地图和OSM地图切换
 * 2. 全屏地图展示
 * 3. 加载并展示GeoJSON格式的农田用水数据
 * 4. 蓝色系渐变色彩表达用水量差异
 * 5. 鼠标悬停显示详细信息
 *
 * 依赖库：OpenLayers
 * 作者 ：学GIS的小宝同学
 */

// OpenLayers核心组件导入
import Map from 'ol/Map.js';                    // 地图主类
import View from 'ol/View.js';                  // 地图视图
import Attribution from 'ol/control/Attribution.js'; // 版权信息控件
import {defaults as defaultControls} from 'ol/control/defaults.js'; // 默认控件
import TileLayer from 'ol/layer/Tile.js';       // 瓦片图层
import OSM from 'ol/source/OSM.js';             // OSM数据源
import VectorLayer from 'ol/layer/Vector.js';   // 矢量图层
import VectorSource from 'ol/source/Vector.js'; // 矢量数据源
import GeoJSON from 'ol/format/GeoJSON.js';     // GeoJSON格式解析器
import {fromLonLat} from 'ol/proj.js';         // 坐标转换工具
import {Style, Fill, Stroke} from 'ol/style.js'; // 样式相关类
import XYZ from 'ol/source/XYZ.js';             // XYZ瓦片数据源

// 创建版权信息控件（设置为不可折叠）
const attribution = new Attribution({
  collapsible: false,
});

// ================ 地图图层配置 ================

/**
 * 创建天地图矢量图层
 * 使用天地图WMTS服务，包含道路、水系、政区等基础地理信息
 */
const tiandituLayer = new TileLayer({
  source: new XYZ({
    url: 'http://t0.tianditu.gov.cn/vec_w/wmts?layer=vec&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=f4d0553a23372a2f48c74851c7e46f4d',
  }),
  visible: false, // 初始状态为隐藏
});

/**
 * 创建天地图注记图层
 * 提供中文地名标注，包括行政区划、道路名称、重要地标等
 */
const tiandituLabelLayer = new TileLayer({
  source: new XYZ({
    url: 'http://t0.tianditu.gov.cn/cva_w/wmts?layer=cva&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=f4d0553a23372a2f48c74851c7e46f4d',
  }),
  visible: false, // 初始状态为隐藏
});

/**
 * 创建OSM图层
 * OpenStreetMap开源地图，默认显示
 */
const osmLayer = new TileLayer({
  source: new OSM(),
  visible: true, // 初始状态为显示
});

/**
 * 创建矢量图层用于显示农田用水数据
 * 通过样式函数实现根据用水量的颜色渐变效果
 */
const vectorLayer = new VectorLayer({
  source: new VectorSource(),
  style: function(feature) {
    const waterUsage = feature.get('农田用水');
    return new Style({
      fill: new Fill({
        color: getWaterColor(waterUsage), // 根据用水量获取颜色
      }),
      stroke: new Stroke({
        color: '#333333', // 边框颜色
        width: 1,          // 边框宽度
      }),
    });
  },
});

// ================ 样式函数 ================

/**
 * 根据农田用水量获取对应的颜色
 * @param {number} waterUsage - 农田用水量（万立方米）
 * @returns {string} 返回对应的RGBA颜色值
 *
 * 颜色渐变逻辑：
 * - 无数据：浅灰色 rgba(230, 240, 255, 0.4)
 * - 低用水量：浅蓝色
 * - 高用水量：深蓝色
 * - 透明度固定为0.4，保证底图可见性
 */
function getWaterColor(waterUsage) {
  if (!waterUsage) return 'rgba(230, 240, 255, 0.4)';

  // 计算颜色渐变（由浅入深）
  const maxUsage = 5000; // 假设最大用水量
  const ratio = Math.min(waterUsage / maxUsage, 1);

  // 蓝色系渐变：从浅蓝到深蓝
  const r = Math.floor(230 - ratio * 150); // 230 -> 80
  const g = Math.floor(240 - ratio * 140); // 240 -> 100
  const b = Math.floor(255 - ratio * 100); // 255 -> 155

  return `rgba(${r}, ${g}, ${b}, 0.4)`;
}

// ================ 地图初始化 ================

/**
 * 创建地图实例
 * 配置图层、控件和视图
 */
const map = new Map({
  layers: [tiandituLayer, tiandituLabelLayer, osmLayer, vectorLayer], // 图层顺序
  controls: defaultControls({attribution: false}).extend([attribution]), // 控件配置
  target: 'map', // 挂载到DOM元素
  view: new View({
    center: fromLonLat([107.5, 34.5]), // 陕西省中心位置坐标
    zoom: 7, // 初始缩放级别
  }),
});

// ================ 数据加载 ================

/**
 * 加载GeoJSON格式的农田用水数据
 * 从本地文件读取数据并添加到矢量图层
 */
fetch('./data.geojson')
  .then(response => response.json())
  .then(data => {
    // 解析GeoJSON数据，转换为OpenLayers要素
    const features = new GeoJSON().readFeatures(data, {
      featureProjection: 'EPSG:3857' // 指定投影坐标系
    });

    // 将要素添加到矢量图层
    vectorLayer.getSource().addFeatures(features);

    // 自动调整视图以适应所有要素
    const extent = vectorLayer.getSource().getExtent();
    map.getView().fit(extent, {
      padding: [50, 50, 50, 50], // 设置内边距
      duration: 1000 // 动画持续时间（毫秒）
    });
  })
  .catch(error => {
    console.error('加载GeoJSON数据失败:', error);
  });

// ================ 地图切换功能 ================

/**
 * 切换地图图层显示状态
 * @param {string} mapType - 地图类型 ('tianditu' 或 'osm')
 */
function switchMapLayer(mapType) {
  if (mapType === 'tianditu') {
    // 显示天地图（矢量+注记）
    tiandituLayer.setVisible(true);
    tiandituLabelLayer.setVisible(true);
    osmLayer.setVisible(false);
  } else {
    // 显示OSM地图
    tiandituLayer.setVisible(false);
    tiandituLabelLayer.setVisible(false);
    osmLayer.setVisible(true);
  }
}

// 加载GeoJSON数据
fetch('./data.geojson')
  .then(response => response.json())
  .then(data => {
    const features = new GeoJSON().readFeatures(data, {
      featureProjection: 'EPSG:3857'
    });
    vectorLayer.getSource().addFeatures(features);

    // 自动调整视图以适应所有要素
    const extent = vectorLayer.getSource().getExtent();
    map.getView().fit(extent, {
      padding: [50, 50, 50, 50],
      duration: 1000
    });
  })
  .catch(error => {
    console.error('加载GeoJSON数据失败:', error);
  });

// 地图切换功能
function switchMapLayer(mapType) {
  if (mapType === 'tianditu') {
    tiandituLayer.setVisible(true);
    tiandituLabelLayer.setVisible(true);
    osmLayer.setVisible(false);
  } else {
    tiandituLayer.setVisible(false);
    tiandituLabelLayer.setVisible(false);
    osmLayer.setVisible(true);
  }
}

// 创建地图切换控件
function createMapSwitcher() {
  const switcher = document.createElement('div');
  switcher.className = 'map-switcher';
  switcher.style.cssText = `
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 1000;
    background: white;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    padding: 5px;
  `;

  const osmButton = document.createElement('button');
  osmButton.textContent = 'OSM';
  osmButton.style.cssText = `
    margin: 2px;
    padding: 5px 10px;
    border: 1px solid #ccc;
    border-radius: 3px;
    background: ${osmLayer.getVisible() ? '#007bff' : 'white'};
    color: ${osmLayer.getVisible() ? 'white' : 'black'};
    cursor: pointer;
  `;

  const tiandituButton = document.createElement('button');
  tiandituButton.textContent = '天地图';
  tiandituButton.style.cssText = `
    margin: 2px;
    padding: 5px 10px;
    border: 1px solid #ccc;
    border-radius: 3px;
    background: ${tiandituLayer.getVisible() ? '#007bff' : 'white'};
    color: ${tiandituLayer.getVisible() ? 'white' : 'black'};
    cursor: pointer;
  `;

  osmButton.addEventListener('click', () => {
    switchMapLayer('osm');
    updateButtonStyles();
  });

  tiandituButton.addEventListener('click', () => {
    switchMapLayer('tianditu');
    updateButtonStyles();
  });

  function updateButtonStyles() {
    osmButton.style.background = osmLayer.getVisible() ? '#007bff' : 'white';
    osmButton.style.color = osmLayer.getVisible() ? 'white' : 'black';
    tiandituButton.style.background = tiandituLayer.getVisible() ? '#007bff' : 'white';
    tiandituButton.style.color = tiandituLayer.getVisible() ? 'white' : 'black';
  }

  switcher.appendChild(osmButton);
  switcher.appendChild(tiandituButton);

  document.getElementById('map').appendChild(switcher);
}

// 全屏功能
function toggleFullscreen() {
  const mapElement = document.getElementById('map');

  if (!document.fullscreenElement) {
    mapElement.requestFullscreen().catch(err => {
      console.error('无法进入全屏模式:', err);
    });
  } else {
    document.exitFullscreen();
  }
}

// 创建全屏按钮
function createFullscreenButton() {
  const button = document.createElement('button');
  button.textContent = '全屏';
  button.style.cssText = `
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 1000;
    padding: 8px 16px;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    cursor: pointer;
    font-size: 14px;
  `;

  button.addEventListener('click', toggleFullscreen);

  document.getElementById('map').appendChild(button);
}

// 创建信息提示框
function createTooltip() {
  const tooltip = document.createElement('div');
  tooltip.className = 'ol-tooltip';
  tooltip.style.cssText = `
    position: absolute;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    pointer-events: none;
    z-index: 1000;
    display: none;
    max-width: 200px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  `;

  document.getElementById('map').appendChild(tooltip);
  return tooltip;
}

// 创建鼠标悬停交互
function createHoverInteraction() {
  const tooltip = createTooltip();
  let highlightFeature = null;

  // 鼠标移动事件
  map.on('pointermove', function(evt) {
    const pixel = evt.pixel;
    const feature = map.forEachFeatureAtPixel(pixel, function(feature) {
      return feature;
    });

    if (feature && feature.get('农田用水')) {
      const waterUsage = feature.get('农田用水');
      const name = feature.get('name') || '未知区域';

      tooltip.innerHTML = `
        <div><strong>${name}</strong></div>
        <div>农田用水: ${waterUsage} 万立方米</div>
        <div>行政区划代码: ${feature.get('adcode') || '无'}</div>
      `;

      tooltip.style.display = 'block';
      tooltip.style.left = (pixel[0] + 10) + 'px';
      tooltip.style.top = (pixel[1] - 10) + 'px';

      // 高亮效果
      if (highlightFeature !== feature) {
        // 重置之前的高亮
        if (highlightFeature) {
          highlightFeature.setStyle(null);
        }

        // 设置新的高亮样式
        feature.setStyle(new Style({
          fill: new Fill({
            color: getWaterColor(feature.get('农田用水')).replace('0.4', '0.6'),
          }),
          stroke: new Stroke({
            color: '#ff6b35',
            width: 2,
          }),
        }));

        highlightFeature = feature;
      }
    } else {
      tooltip.style.display = 'none';

      // 重置高亮
      if (highlightFeature) {
        highlightFeature.setStyle(null);
        highlightFeature = null;
      }
    }
  });

  // 鼠标离开地图区域
  map.getTargetElement().addEventListener('mouseleave', function() {
    tooltip.style.display = 'none';

    // 重置高亮
    if (highlightFeature) {
      highlightFeature.setStyle(null);
      highlightFeature = null;
    }
  });
}

// 初始化控件
createMapSwitcher();
createFullscreenButton();
createHoverInteraction();

// 设置地图全屏样式
document.getElementById('map').style.cssText = `
  width: 100vw;
  height: 100vh;
  position: fixed;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
`;

function checkSize() {
  const small = map.getSize()[0] < 600;
  attribution.setCollapsible(small);
  attribution.setCollapsed(small);
}

map.on('change:size', checkSize);
checkSize();
