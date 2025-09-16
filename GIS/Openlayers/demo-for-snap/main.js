/**
 * 天地图卫星影像 + OpenLayers自定义线段捕捉功能示例
 *
 * 功能特点：
 * - 加载天地图卫星影像底图
 * - 支持绘制线条要素
 * - 支持要素编辑修改
 * - 自定义线段捕捉功能，支持端点和中点捕捉
 * - 全屏显示地图
 *
 * @author 学GIS的小宝同学
 * @version 1.0.0
 * @date 2025
 */

// 导入OpenLayers核心组件
import Map from 'ol/Map.js';                    // 地图主类
import View from 'ol/View.js';                  // 地图视图
import Draw from 'ol/interaction/Draw.js';      // 绘制交互
import Modify from 'ol/interaction/Modify.js';  // 修改交互
import Snap from 'ol/interaction/Snap.js';      // 捕捉交互
import VectorLayer from 'ol/layer/Vector.js';    // 矢量图层
import VectorSource from 'ol/source/Vector.js';  // 矢量数据源
import TileLayer from 'ol/layer/Tile.js';       // 瓦片图层
import XYZ from 'ol/source/XYZ.js';              // XYZ数据源
import { fromLonLat } from 'ol/proj.js';        // 坐标转换工具

// 天地图访问令牌
const tiandituToken = 'f4d0553a23372a2f48c74851c7e46f4d';

// 创建天地图卫星影像图层
// 使用XYZ数据源加载天地图瓦片服务
const tiandituLayer = new TileLayer({
  source: new XYZ({
    // 天地图卫星影像URL模板，支持多服务器负载均衡
    url: 'http://t{0-7}.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=' + tiandituToken,
  }),
});

// 创建矢量图层用于绘制要素
// 该图层将用于存储用户绘制的线条和其他矢量要素
const vector = new VectorLayer({
  source: new VectorSource(),
  style: {
    'stroke-color': '#ffcc33',  // 设置线条颜色为金黄色
  },
});

// 初始化地图实例
// 配置地图图层、目标容器和视图
const map = new Map({
  layers: [tiandituLayer, vector],  // 图层顺序：卫星图层在底部，矢量图层在顶部
  target: 'map',                    // 地图容器div的id
  view: new View({
    center: fromLonLat([108.948024, 34.263161]),  // 陕西省西安市坐标
    zoom: 7,                            // 缩放级别
  }),
});

// 创建修改交互
// 允许用户编辑已绘制的要素
const modify = new Modify({
  source: vector.getSource(),  // 作用在矢量图层的要素上
});
map.addInteraction(modify);

// 创建绘制交互
// 允许用户绘制线条要素
const draw = new Draw({
  source: vector.getSource(),  // 将绘制的要素添加到矢量图层
  type: 'LineString',          // 绘制类型为线
});
map.addInteraction(draw);

// 创建捕捉交互
// 实现自定义的线段捕捉功能
const snap = new Snap({
  source: vector.getSource(),  // 捕捉矢量图层中的要素
  segmenters: {
    // 自定义线段分割器
    // 将每条线段分割为多个可捕捉的点
    LineString: (geometry) => {
      const segments = [];
      geometry.forEachSegment((c1, c2) => {
        // 添加原始线段的两个端点
        segments.push([c1, c2]);
        // 添加线段的中点作为额外的捕捉点
        segments.push([[(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2]]);
      });
      return segments;
    },
  },
});
map.addInteraction(snap);
