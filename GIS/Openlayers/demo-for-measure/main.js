/**
 * OpenLayers测量工具演示程序
 * 作者：学GIS的小宝同学
 * 功能：支持距离测量和面积测量，使用天地图作为底图
 *
 * 本程序展示了如何使用OpenLayers实现地图测量功能，包括：
 * 1. 底图加载和图层管理
 * 2. 交互式绘制功能
 * 3. 距离和面积计算
 * 4. 用户界面设计和事件处理
 * 5. 测量结果的显示和管理
 */

// 导入OpenLayers核心模块
import Map from 'ol/Map.js';                    // 地图核心类
import {unByKey} from 'ol/Observable.js';       // 事件监听器移除工具
import Overlay from 'ol/Overlay.js';            // 覆盖物类，用于显示提示信息
import View from 'ol/View.js';                  // 地图视图类
import LineString from 'ol/geom/LineString.js'; // 线几何体类
import Polygon from 'ol/geom/Polygon.js';       // 多边形几何体类
import Draw from 'ol/interaction/Draw.js';      // 绘制交互类
import TileLayer from 'ol/layer/Tile.js';       // 瓦片图层类
import VectorLayer from 'ol/layer/Vector.js';   // 矢量图层类
import XYZ from 'ol/source/XYZ.js';             // XYZ数据源类
import VectorSource from 'ol/source/Vector.js'; // 矢量数据源类
import {getArea, getLength} from 'ol/sphere.js'; // 球面测量工具
import CircleStyle from 'ol/style/Circle.js';    // 圆形样式类
import Fill from 'ol/style/Fill.js';            // 填充样式类
import Stroke from 'ol/style/Stroke.js';         // 描边样式类
import Style from 'ol/style/Style.js';           // 样式类
import { fromLonLat } from 'ol/proj.js';        // 坐标转换工具

/**
 * 创建天地图影像底图图层
 * 使用天地图DataServer接口加载影像瓦片数据
 * URL参数说明：
 * - T=img_w: 影像图（墨卡托投影）
 * - x,y,l: 瓦片坐标和缩放级别
 * - tk: 天地图API密钥
 */
const raster = new TileLayer({
  source: new XYZ({
    url: 'http://t{0-7}.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=f4d0553a23372a2f48c74851c7e46f4d',
    wrapX: false  // 禁用水平重复，防止地图在180度经线处重复显示
  })
});

/**
 * 创建天地图影像注记图层
 * 使用WMTS标准接口加载注记瓦片数据
 * 注记图层包含地名、道路名等文字标注
 */
const annotation = new TileLayer({
  source: new XYZ({
    url: 'https://t{0-7}.tianditu.gov.cn/cia_w/wmts?layer=cia&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=f4d0553a23372a2f48c74851c7e46f4d',
    wrapX: false
  })
});

/**
 * 创建矢量数据源
 * 用于存储用户绘制的测量要素（线和多边形）
 * VectorSource是OpenLayers中存储矢量要素的核心类
 */
const source = new VectorSource();

/**
 * 创建矢量图层
 * 用于显示用户绘制的测量要素
 * 配置了要素的默认样式：半透明填充、黄色描边、圆形节点
 */
const vector = new VectorLayer({
  source: source,  // 关联矢量数据源
  style: {
    'fill-color': 'rgba(255, 255, 255, 0.2)',      // 多边形填充颜色（半透明白色）
    'stroke-color': '#ffcc33',                     // 描边颜色（黄色）
    'stroke-width': 2,                             // 描边宽度
    'circle-radius': 7,                            // 圆形节点半径
    'circle-fill-color': '#ffcc33',                // 圆形节点填充颜色
  },
});

/**
 * 当前正在绘制的要素
 * 在用户绘制过程中实时更新，用于预览效果
 * @type {import('ol/Feature.js').default}
 */
let sketch;

/**
 * 帮助提示框DOM元素
 * 用于显示绘制过程中的操作提示信息
 * @type {HTMLElement}
 */
let helpTooltipElement;

/**
 * 帮助提示框覆盖物
 * 将DOM元素显示在地图指定位置上
 * @type {Overlay}
 */
let helpTooltip;

/**
 * 测量结果提示框DOM元素
 * 用于显示距离或面积的测量结果
 * @type {HTMLElement}
 */
let measureTooltipElement;

/**
 * 测量结果提示框覆盖物
 * 将测量结果显示在地图上的对应位置
 * @type {Overlay}
 */
let measureTooltip;

/**
 * 绘制多边形时的提示信息
 * 提示用户如何继续绘制多边形
 * @type {string}
 */
const continuePolygonMsg = '点击继续绘制多边形';

/**
 * 绘制线段时的提示信息
 * 提示用户如何继续绘制线段
 * @type {string}
 */
const continueLineMsg = '点击继续绘制线段';

/**
 * 鼠标移动事件处理函数
 * 根据当前绘制状态动态更新帮助提示信息
 * @param {import('../src/ol/MapBrowserEvent').default} evt 地图浏览器事件
 */
const pointerMoveHandler = function (evt) {
  // 如果正在拖动地图，不显示提示信息
  if (evt.dragging) {
    return;
  }

  // 设置默认提示信息
  let helpMsg = '点击开始绘制';

  // 根据当前绘制的几何类型更新提示信息
  if (sketch) {
    const geom = sketch.getGeometry();
    if (geom instanceof Polygon) {
      helpMsg = continuePolygonMsg;  // 多边形绘制提示
    } else if (geom instanceof LineString) {
      helpMsg = continueLineMsg;    // 线段绘制提示
    }
  }

  // 更新提示框内容和位置
  helpTooltipElement.innerHTML = helpMsg;
  helpTooltip.setPosition(evt.coordinate);

  // 显示提示框
  helpTooltipElement.classList.remove('hidden');
};

/**
 * 创建地图实例
 * 配置地图图层、目标容器和视图参数
 *
 * 图层顺序：影像底图 -> 影像注记 -> 矢量测量图层
 * 中心点：西安市（经度108.948024，纬度34.263161）
 * 缩放级别：7（适合查看陕西省范围）
 */
const map = new Map({
  layers: [raster, annotation, vector],  // 图层列表，按顺序叠加
  target: 'map',                           // 地图容器的DOM元素ID
  view: new View({
    center: fromLonLat([108.948024, 34.263161]),  // 西安市坐标（经纬度转墨卡托投影）
    zoom: 7,                            // 初始缩放级别
  }),
});

// 为地图添加鼠标移动事件监听器
map.on('pointermove', pointerMoveHandler);

// 为地图视口添加鼠标离开事件监听器
// 当鼠标离开地图区域时隐藏帮助提示框
map.getViewport().addEventListener('mouseout', function () {
  helpTooltipElement.classList.add('hidden');
});

/**
 * 获取界面控制元素
 */
const lengthButton = document.getElementById('measure-length');    // 测距按钮
const areaButton = document.getElementById('measure-area');        // 测面积按钮
const clearButton = document.getElementById('clear-button');      // 清除按钮

/**
 * 绘制交互对象
 * 全局变量，用于在需要时移除交互功能
 */
let draw;

/**
 * 格式化长度输出
 * 将计算得到的长度根据大小转换为合适的单位（米或千米）
 *
 * @param {LineString} line 线几何体对象
 * @return {string} 格式化后的长度字符串
 */
const formatLength = function (line) {
  // 使用球面测量方法计算线长度（单位：米）
  const length = getLength(line);
  let output;

  // 长度大于100米时转换为千米
  if (length > 100) {
    output = Math.round((length / 1000) * 100) / 100 + ' ' + 'km';
  } else {
    output = Math.round(length * 100) / 100 + ' ' + 'm';
  }
  return output;
};

/**
 * 格式化面积输出
 * 将计算得到的面积根据大小转换为合适的单位（平方米或平方千米）
 *
 * @param {Polygon} polygon 多边形几何体对象
 * @return {string} 格式化后的面积字符串
 */
const formatArea = function (polygon) {
  // 使用球面测量方法计算多边形面积（单位：平方米）
  const area = getArea(polygon);
  let output;

  // 面积大于10000平方米时转换为平方千米
  if (area > 10000) {
    output = Math.round((area / 1000000) * 100) / 100 + ' ' + 'km<sup>2</sup>';
  } else {
    output = Math.round(area * 100) / 100 + ' ' + 'm<sup>2</sup>';
  }
  return output;
};

/**
 * 绘制过程中的要素样式
 * 用于显示用户正在绘制的要素预览效果
 *
 * 样式特点：
 * - 半透明白色填充
 * - 红色虚线描边（橡皮筋效果）
 * - 圆形节点用于多边形顶点
 */
const style = new Style({
  fill: new Fill({
    color: 'rgba(255, 255, 255, 0.2)',  // 半透明白色填充
  }),
  stroke: new Stroke({
    color: 'red',                        // 红色描边（橡皮筋效果）
    lineDash: [10, 10],                  // 虚线样式
    width: 2,                             // 线宽
  }),
  image: new CircleStyle({               // 圆形节点样式
    radius: 5,                            // 半径
    stroke: new Stroke({
      color: 'rgba(0, 0, 0, 0.7)',       // 描边颜色
    }),
    fill: new Fill({
      color: 'rgba(255, 255, 255, 0.2)', // 填充颜色
    }),
  }),
});

/**
 * 添加绘制交互
 * 根据指定的几何类型创建绘制交互对象，并设置相关事件处理
 *
 * @param {string} type 几何类型（'LineString' 或 'Polygon'）
 */
function addInteraction(type) {
  // 创建绘制交互对象
  draw = new Draw({
    source: source,                    // 矢量数据源，用于存储绘制的要素
    type: type,                        // 几何类型
    style: function (feature) {        // 动态样式函数
      const geometryType = feature.getGeometry().getType();
      // 只为当前绘制的几何类型和顶点应用样式
      if (geometryType === type || geometryType === 'Point') {
        return style;
      }
    },
  });

  // 将交互添加到地图
  map.addInteraction(draw);

  // 创建测量结果提示框和帮助提示框
  createMeasureTooltip();
  createHelpTooltip();

  let listener;
  // 监听绘制开始事件
  draw.on('drawstart', function (evt) {
    // 设置当前绘制的要素
    sketch = evt.feature;

    let tooltipCoord;

    // 监听几何体变化事件，实时更新测量结果
    listener = sketch.getGeometry().on('change', function (evt) {
      const geom = evt.target;  // 当前几何体
      let output;

      // 根据几何类型计算相应的测量值
      if (geom instanceof Polygon) {
        output = formatArea(geom);                           // 计算面积
        tooltipCoord = geom.getInteriorPoint().getCoordinates(); // 获取多边形中心点
      } else if (geom instanceof LineString) {
        output = formatLength(geom);                          // 计算长度
        tooltipCoord = geom.getLastCoordinate();              // 获取线段终点
      }

      // 更新测量结果显示
      measureTooltipElement.innerHTML = output;
      measureTooltip.setPosition(tooltipCoord);
    });
  });

  // 监听绘制结束事件
  draw.on('drawend', function () {
    // 将测量结果提示框改为静态显示
    measureTooltipElement.className = 'ol-tooltip ol-tooltip-static';
    measureTooltip.setOffset([0, -7]);

    // 清理绘制状态
    sketch = null;                   // 清除当前绘制的要素
    measureTooltipElement = null;    // 清除测量提示框元素
    createMeasureTooltip();           // 创建新的测量提示框以备下次使用
    unByKey(listener);                // 移除几何体变化监听器
  });
}

/**
 * 创建帮助提示框
 * 用于显示绘制过程中的操作指导信息
 * 每次激活测量功能时都会创建新的提示框
 */
function createHelpTooltip() {
  // 如果已存在帮助提示框，先移除
  if (helpTooltipElement) {
    helpTooltipElement.remove();
  }

  // 创建新的提示框DOM元素
  helpTooltipElement = document.createElement('div');
  helpTooltipElement.className = 'ol-tooltip hidden';

  // 创建覆盖物对象，将提示框显示在地图上
  helpTooltip = new Overlay({
    element: helpTooltipElement,     // DOM元素
    offset: [15, 0],                // 偏移量（向右偏移15像素）
    positioning: 'center-left',     // 定位方式（相对于鼠标位置居左）
  });

  // 将覆盖物添加到地图
  map.addOverlay(helpTooltip);
}

/**
 * 创建测量结果提示框
 * 用于实时显示距离或面积的测量结果
 * 每次开始新的测量时都会创建新的提示框
 */
function createMeasureTooltip() {
  // 如果已存在测量提示框，先移除
  if (measureTooltipElement) {
    measureTooltipElement.remove();
  }

  // 创建新的测量提示框DOM元素
  measureTooltipElement = document.createElement('div');
  measureTooltipElement.className = 'ol-tooltip ol-tooltip-measure';

  // 创建覆盖物对象，将测量结果显示在地图上
  measureTooltip = new Overlay({
    element: measureTooltipElement,   // DOM元素
    offset: [0, -15],                 // 偏移量（向上偏移15像素）
    positioning: 'bottom-center',      // 定位方式（相对于坐标点居下）
    stopEvent: false,                 // 不阻止事件传播
    insertFirst: false,               // 不插入到其他元素之前
  });

  // 将覆盖物添加到地图
  map.addOverlay(measureTooltip);
}

/**
 * 长度测量按钮点击事件处理
 * 用户点击测距按钮时切换到长度测量模式
 */
lengthButton.addEventListener('click', function() {
  // 移除现有的绘制交互，避免冲突
  map.removeInteraction(draw);

  // 更新按钮状态，视觉上显示当前激活的功能
  lengthButton.classList.add('active');
  areaButton.classList.remove('active');

  // 添加长度测量交互，支持线段绘制
  addInteraction('LineString');
});

/**
 * 面积测量按钮点击事件处理
 * 用户点击测面积按钮时切换到面积测量模式
 */
areaButton.addEventListener('click', function() {
  // 移除现有的绘制交互，避免冲突
  map.removeInteraction(draw);

  // 更新按钮状态，视觉上显示当前激活的功能
  areaButton.classList.add('active');
  lengthButton.classList.remove('active');

  // 添加面积测量交互，支持多边形绘制
  addInteraction('Polygon');
});

/**
 * 按钮Tooltip交互功能
 * 为所有工具按钮添加鼠标悬停提示功能
 */
const tooltip = document.getElementById('tooltip');
const buttons = document.querySelectorAll('.tool-button');

// 为每个按钮添加鼠标事件监听
buttons.forEach(button => {
  // 鼠标进入时显示提示信息
  button.addEventListener('mouseenter', function(e) {
    const tooltipText = this.getAttribute('data-tooltip');
    tooltip.textContent = tooltipText;
    tooltip.classList.add('show');

    // 动态计算提示框位置，确保显示在按钮上方
    const rect = this.getBoundingClientRect();
    tooltip.style.left = (rect.left + rect.width / 2 - tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 8) + 'px';
  });

  // 鼠标离开时隐藏提示信息
  button.addEventListener('mouseleave', function() {
    tooltip.classList.remove('show');
  });
});

/**
 * 清除按钮点击事件处理
 * 用户点击清除按钮时清理所有绘制痕迹和测量结果
 */
clearButton.addEventListener('click', function() {
  // 清除所有绘制的要素（线和多边形）
  source.clear();

  // 移除所有静态测量结果提示框
  const tooltips = document.querySelectorAll('.ol-tooltip-static');
  tooltips.forEach(tooltip => tooltip.remove());

  // 重置当前绘制状态
  if (draw) {
    map.removeInteraction(draw);
    draw = null;
  }

  // 重置按钮状态，取消所有按钮的激活状态
  lengthButton.classList.remove('active');
  areaButton.classList.remove('active');
  sketch = null;
});

/**
 * 程序初始化
 * 设置默认的测量模式为长度测量
 */
addInteraction('LineString');
lengthButton.classList.add('active');
