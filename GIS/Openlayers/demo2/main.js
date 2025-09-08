import Map from 'ol/Map.js';
import View from 'ol/View.js';
import Draw from 'ol/interaction/Draw.js';
import TileLayer from 'ol/layer/Tile.js';
import VectorLayer from 'ol/layer/Vector.js';
import OSM from 'ol/source/OSM.js';
import VectorSource from 'ol/source/Vector.js';
import {fromLonLat} from 'ol/proj.js';

/**
 * 教学 Demo：OpenLayers 绘制要素 + 自定义样式
 *
 * 本示例演示：
 * 1) 如何创建一个基础地图（OSM 瓦片图层）
 * 2) 如何创建一个用于存放绘制结果的矢量数据源与图层
 * 3) 如何让地图支持交互式绘制（点、线、面、圆）
 * 4) 如何为不同几何类型设置不同的样式
 * 5) 如何通过下拉框切换交互绘制的几何类型
 *
 * 使用方式：
 * - 打开页面后，左上角控制面板选择要绘制的几何类型，然后在地图上点击/拖拽绘制。
 * - 切换类型会移除旧的绘制交互并重新添加新的交互。
 */

// 1) 创建底图图层：使用 OpenStreetMap 作为瓦片底图
const raster = new TileLayer({
  source: new OSM(),
});

// 2) 创建矢量数据源：用于存放用户绘制的几何
// wrapX: false 表示不要在国际换日线附近复制要素
const source = new VectorSource({wrapX: false});

// 3) 创建矢量图层：把上述数据源放入图层，供地图渲染
const vector = new VectorLayer({
  source: source,
});

// 4) 创建地图实例，将底图与矢量图层添加到地图上
const map = new Map({
  layers: [raster, vector], // 图层渲染顺序从下到上
  target: 'map', // 绑定到 index.html 中 id 为 "map" 的容器
  view: new View({
    // 将地图中心设置为中国大致中心（经纬度 -> Web Mercator）
    center: fromLonLat([104.195397, 35.86166]),
    zoom: 4, // 初始缩放级别
  }),
});

// 5) 为不同几何类型定义样式（简化写法，OpenLayers 会将这些键映射到真实样式）
// 说明：
// - circle-radius / circle-fill-color 控制点样式（圆形点）
// - stroke-color / stroke-width 控制线或面的边框样式
// - fill-color 控制面的填充颜色
const styles = {
  Point: {
    'circle-radius': 5,
    'circle-fill-color': 'red',
  },
  LineString: {
    'circle-radius': 5,
    'circle-fill-color': 'red',
    'stroke-color': 'yellow',
    'stroke-width': 2,
  },
  Polygon: {
    'circle-radius': 5,
    'circle-fill-color': 'red',
    'stroke-color': 'yellow',
    'stroke-width': 2,
    'fill-color': 'blue',
  },
  Circle: {
    'circle-radius': 5,
    'circle-fill-color': 'red',
    'stroke-color': 'blue',
    'stroke-width': 2,
    'fill-color': 'yellow',
  },
};

// 6) 获取下拉选择框，用于切换绘制类型
const typeSelect = document.getElementById('type');

// 保留对当前绘制交互的引用，以便后续移除
let draw; // global so we can remove it later

/**
 * 根据下拉框当前值，创建并添加对应的绘制交互到地图。
 * - 当选择为 'None' 时，不添加任何绘制交互。
 */
function addInteraction() {
  const value = typeSelect.value;
  if (value !== 'None') {
    draw = new Draw({
      source: source, // 绘制产生的几何会自动添加到该数据源
      type: typeSelect.value, // 几何类型：Point / LineString / Polygon / Circle
      style: styles[value], // 应用对应的样式定义
    });
    map.addInteraction(draw);
  }
}

/**
 * 处理下拉框的 change 事件：
 * - 移除旧的绘制交互
 * - 按照新的类型重新添加绘制交互
 */
typeSelect.onchange = function () {
  map.removeInteraction(draw);
  addInteraction();
};

// 初次进入页面时，按当前下拉框默认值添加一次绘制交互
addInteraction();
