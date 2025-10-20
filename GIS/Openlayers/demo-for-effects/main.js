/**
 * OpenLayers地图效果演示项目
 * 作者：学GIS的小宝同学
 * 功能：实现地图图层切换和动态点闪烁动画效果
 *
 * 本项目展示了OpenLayers的核心功能：
 * 1. 多种地图源的使用（OSM、天地图）
 * 2. 图层管理和切换
 * 3. 矢量图层和动画效果
 * 4. 事件监听和交互
 */

// ==================== 导入OpenLayers核心模块 ====================
// Feature: 地理要素类，用于表示地图上的单个要素（如点、线、面）
import Feature from 'ol/Feature.js';
// Map: 地图主类，是OpenLayers的核心组件
import Map from 'ol/Map.js';
// unByKey: 用于移除事件监听器的工具函数
import {unByKey} from 'ol/Observable.js';
// View: 地图视图类，控制地图的显示范围、缩放级别等
import View from 'ol/View.js';
// easeOut: 缓动函数，用于创建平滑的动画效果
import {easeOut} from 'ol/easing.js';
// Point: 点几何类，用于表示点状要素
import Point from 'ol/geom/Point.js';
// TileLayer: 瓦片图层类，用于显示瓦片地图（如OSM、天地图）
import TileLayer from 'ol/layer/Tile.js';
// VectorLayer: 矢量图层类，用于显示矢量数据（点、线、面）
import VectorLayer from 'ol/layer/Vector.js';
// fromLonLat: 坐标转换函数，将经纬度转换为投影坐标
import {fromLonLat} from 'ol/proj.js';
// getVectorContext: 获取矢量上下文，用于在渲染过程中绘制矢量图形
import {getVectorContext} from 'ol/render.js';
// OSM: OpenStreetMap数据源类
import OSM from 'ol/source/OSM.js';
// VectorSource: 矢量数据源类，用于管理矢量要素
import VectorSource from 'ol/source/Vector.js';
// CircleStyle: 圆形样式类，用于点状要素的样式
import CircleStyle from 'ol/style/Circle.js';
// Stroke: 线条样式类，用于定义线条的颜色、宽度等
import Stroke from 'ol/style/Stroke.js';
// Style: 样式类，用于定义要素的完整样式
import Style from 'ol/style/Style.js';
// XYZ: XYZ数据源类，用于访问标准的瓦片地图服务
import XYZ from 'ol/source/XYZ.js';
// LayerGroup: 图层组类，用于将多个图层组合为一个逻辑组
import {Group as LayerGroup} from 'ol/layer.js';

// ==================== 地图图层配置 ====================

// 创建OSM（OpenStreetMap）图层
// OSM是开源的免费地图服务，适合学习和演示使用
const osmLayer = new TileLayer({
  source: new OSM({
    wrapX: false,  // 禁用X方向环绕，防止地图在全球范围内重复显示
  }),
});

// 天地图API密钥 - 用于访问天地图服务
// 注意：实际项目中请保护好自己的密钥，不要暴露在客户端代码中
const tiandituKey = 'f4d0553a23372a2f48c74851c7e46f4d';

// 创建天地图影像图层
// 天地图是中国国家地理信息公共服务平台提供的地图服务
// img_w表示Web墨卡托投影的影像图层
const tiandituImageLayer = new TileLayer({
  source: new XYZ({
    // WMTS服务URL，天地图使用标准的WMTS（Web Map Tile Service）接口
    // t{0-7}表示使用8个子域名进行负载均衡，提高访问速度
    // {z}、{x}、{y}分别是缩放级别、列号、行号
    // tk参数是天地图的API密钥
    url: 'http://t{0-7}.tianditu.gov.cn/img_w/wmts?layer=img&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=' + tiandituKey,
    wrapX: false,  // 禁用X方向环绕
  }),
});

// 创建天地图注记图层
// cia_w表示Web墨卡托投影的中文注记图层
// 注记图层通常叠加在影像图层上，提供地名、道路等信息
const tiandituAnnotationLayer = new TileLayer({
  source: new XYZ({
    url: 'http://t{0-7}.tianditu.gov.cn/cia_w/wmts?layer=cia&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=' + tiandituKey,
    wrapX: false,  // 禁用X方向环绕
  }),
});

// 创建天地图图层组
// 将影像图层和注记图层组合为一个逻辑组，便于统一管理
// 这样切换到天地图时，会同时显示影像和注记
const tiandituGroup = new LayerGroup({
  layers: [tiandituImageLayer, tiandituAnnotationLayer],
});

// ==================== 矢量图层配置 ====================

// 创建矢量数据源
// VectorSource用于管理矢量要素，支持动态添加、删除、修改要素
const source = new VectorSource({
  wrapX: false,  // 禁用X方向环绕
});

// 创建矢量图层
// VectorLayer用于显示矢量数据，可以显示点、线、面等几何类型
// 这里用于显示随机生成的点和闪烁动画效果
const vector = new VectorLayer({
  source: source,  // 指定数据源
});

// ==================== 地图初始化 ====================

// 创建地图实例
// Map是OpenLayers的核心类，用于管理所有图层、交互和控件
const map = new Map({
  layers: [osmLayer, vector],  // 图层列表，按顺序叠加显示
  target: 'map',  // 地图容器的DOM元素ID
  view: new View({
    center: [0, 0],  // 地图中心点坐标（Web墨卡托投影）
    zoom: 1,  // 初始缩放级别
    multiWorld: true,  // 支持多世界显示（当地图缩放很小时显示多个地球）
  }),
});

// ==================== 动态要素生成 ====================

/**
 * 随机生成点要素
 * 在全球范围内随机生成点要素，用于演示动画效果
 */
function addRandomFeature() {
  // 生成随机经纬度坐标
  // 经度范围：-180 到 180，纬度范围：-85 到 85（避免极地地区）
  const x = Math.random() * 360 - 180;
  const y = Math.random() * 170 - 85;

  // 将经纬度坐标转换为Web墨卡托投影坐标
  // OpenLayers内部使用Web墨卡托投影（EPSG:3857）
  const geom = new Point(fromLonLat([x, y]));

  // 创建要素并添加到数据源
  // Feature是地理要素的基本单位，包含几何图形和属性
  const feature = new Feature(geom);
  source.addFeature(feature);
}

// ==================== 动画效果实现 ====================

// 动画持续时间（毫秒）
const duration = 3000;

/**
 * 闪烁动画效果
 * 当新要素添加到地图时，创建一个向外扩散的圆圈动画
 * @param {Feature} feature - 要添加动画效果的要素
 */
function flash(feature) {
  const start = Date.now();  // 记录动画开始时间
  const flashGeom = feature.getGeometry().clone();  // 复制要素几何图形

  // 获取当前基础图层（用于监听渲染事件）
  const currentBaseLayer = map.getLayers().item(0);
  let listenerKey;

  // 根据图层类型选择合适的事件监听器
  // LayerGroup需要监听其内部的具体图层
  if (currentBaseLayer instanceof LayerGroup) {
    listenerKey = currentBaseLayer.getLayers().item(0).on('postrender', animate);
  } else {
    listenerKey = currentBaseLayer.on('postrender', animate);
  }

  /**
   * 动画渲染函数
   * 在每次地图渲染时调用，更新动画效果
   * @param {Object} event - 渲染事件对象
   */
  function animate(event) {
    const frameState = event.frameState;
    const elapsed = frameState.time - start;

    // 检查动画是否完成
    if (elapsed >= duration) {
      unByKey(listenerKey);  // 移除事件监听器，停止动画
      return;
    }

    // 获取矢量渲染上下文，用于在渲染过程中绘制临时图形
    const vectorContext = getVectorContext(event);
    const elapsedRatio = elapsed / duration;  // 计算动画进度（0-1）

    // 使用缓动函数计算动画参数
    // radius: 从5开始，结束于30（5 + 25）
    // opacity: 从1开始，结束于0
    const radius = easeOut(elapsedRatio) * 25 + 5;
    const opacity = easeOut(1 - elapsedRatio);

    // 创建动画样式
    // 使用红色圆圈，随时间逐渐变大并变透明
    const style = new Style({
      image: new CircleStyle({
        radius: radius,  // 圆圈半径
        stroke: new Stroke({
          color: 'rgba(255, 0, 0, ' + opacity + ')',  // 红色，透明度递减
          width: 0.25 + opacity,  // 线条宽度
        }),
      }),
    });

    // 应用样式并绘制几何图形
    vectorContext.setStyle(style);
    vectorContext.drawGeometry(flashGeom);

    // 请求地图继续渲染，以实现动画效果
    // 这会触发下一次的postrender事件
    map.render();
  }
}

// ==================== 事件监听设置 ====================

// 监听要素添加事件
// 当新要素添加到矢量数据源时，自动触发闪烁动画
source.on('addfeature', function (e) {
  flash(e.feature);
});

// 定时添加随机要素
// 每1秒在全球随机位置添加一个点要素，并显示闪烁动画
window.setInterval(addRandomFeature, 1000);

// ==================== 图层切换功能 ====================

// 获取图层切换按钮
const osmBtn = document.getElementById('osm-btn');
const tiandituBtn = document.getElementById('tianditu-btn');

/**
 * OSM地图切换事件处理
 * 点击按钮时切换到OpenStreetMap图层
 */
osmBtn.addEventListener('click', function() {
  // 移除当前的基础图层（索引0）
  map.getLayers().removeAt(0);
  // 在索引0位置插入OSM图层
  map.getLayers().insertAt(0, osmLayer);

  // 更新按钮样式状态
  osmBtn.classList.add('active');
  tiandituBtn.classList.remove('active');
});

/**
 * 天地图切换事件处理
 * 点击按钮时切换到天地图图层组
 */
tiandituBtn.addEventListener('click', function() {
  // 移除当前的基础图层（索引0）
  map.getLayers().removeAt(0);
  // 在索引0位置插入天地图图层组
  map.getLayers().insertAt(0, tiandituGroup);

  // 更新按钮样式状态
  tiandituBtn.classList.add('active');
  osmBtn.classList.remove('active');
});

/**
 * 项目总结：
 *
 * 本项目演示了OpenLayers的核心功能和技术要点：
 *
 * 1. 图层管理：
 *    - 使用TileLayer显示瓦片地图
 *    - 使用VectorLayer显示矢量数据
 *    - 使用LayerGroup组合多个图层
 *    - 动态切换地图图层
 *
 * 2. 数据源：
 *    - OSM开源地图数据源
 *    - XYZ格式的天地图WMTS服务
 *    - VectorSource矢量数据源
 *
 * 3. 动画效果：
 *    - 利用postrender事件实现自定义动画
 *    - 使用缓动函数创建平滑过渡效果
 *    - 通过矢量上下文绘制临时图形
 *
 * 4. 交互功能：
 *    - 按钮点击事件处理
 *    - 图层状态管理
 *    - DOM元素样式控制
 *
 * 5. 坐标系统：
 *    - 经纬度坐标与Web墨卡托投影的转换
 *    - 全球范围坐标生成
 *
 * 这个项目为学习Web GIS开发提供了一个很好的起点！
 */
