/**
 * OpenLayers 地图展示应用 - 农田用水数据可视化
 *
 * 这个应用演示了如何使用OpenLayers创建一个地图展示应用，
 * 包括天地图和OSM地图切换、GeoJSON数据加载、农田用水数据渲染等功能。
 *
 * 主要技术要点：
 * 1. OpenLayers基础架构和使用
 * 2. 多地图源切换功能
 * 3. GeoJSON数据加载和渲染
 * 4. 农田用水数据可视化
 * 5. 全屏地图展示
 */

// ==================== 导入必要的模块 ====================

// OpenLayers核心模块
import Map from 'ol/Map.js';                    // 地图主类
import View from 'ol/View.js';                  // 地图视图类
import Control from 'ol/control/Control.js';    // 控件基类
import {defaults as defaultControls} from 'ol/control/defaults.js'; // 默认控件集合

// 图层相关模块
import TileLayer from 'ol/layer/Tile.js';       // 瓦片图层类
import XYZ from 'ol/source/XYZ.js';             // XYZ瓦片数据源
import VectorLayer from 'ol/layer/Vector.js';   // 矢量图层类
import VectorSource from 'ol/source/Vector.js'; // 矢量数据源
import GeoJSON from 'ol/format/GeoJSON.js';     // GeoJSON数据格式

// 地理要素模块
import {fromLonLat, toLonLat} from 'ol/proj.js'; // 坐标转换函数
import Feature from 'ol/Feature.js';             // 地理要素类
import Polygon from 'ol/geom/Polygon.js';       // 面几何类

// 样式相关模块
import {Style, Fill, Stroke} from 'ol/style.js'; // 样式类

// ==================== 全局变量定义 ====================

// 天地图API密钥 - 用于访问天地图瓦片服务
const tk = 'f4d0553a23372a2f48c74851c7e46f4d';

// 绘制交互相关变量
let drawInteraction;       // 绘制交互对象
let drawLayer;              // 绘制图层
let drawingMode = null;     // 当前绘制模式
let drawSource;             // 绘制数据源

// 范围限制相关变量
let rangeExtent = null;     // 绘制范围限制（经纬度坐标）
let rangeLayer;             // 范围显示图层

// 拓扑检查相关变量
let highlightSource;        // 高亮数据源
let highlightLayer;         // 高亮图层（用于显示拓扑错误）

// ==================== SVG图标定义 ====================

// 工具栏按钮的SVG图标定义
// 每个图标都使用内联SVG格式，便于样式控制
const svgIcons = {
  // 点绘制图标 - 红色圆圈
  point: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="12" cy="12" r="10" fill="red"/>
    <circle cx="12" cy="12" r="3" fill="white"/>
  </svg>`,

  // 线绘制图标 - 蓝色线段
  line: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M4 12 L20 12" stroke="blue"/>
    <circle cx="4" cy="12" r="3" fill="blue"/>
    <circle cx="20" cy="12" r="3" fill="blue"/>
  </svg>`,

  // 面绘制图标 - 绿色多边形
  polygon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M12 2 L22 8 L18 20 L6 20 L2 8 Z" fill="green" fill-opacity="0.3" stroke="green"/>
  </svg>`,

  // 清除图标 - 红色垃圾桶
  clear: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M3 6 L21 6 M19 6 L20 20 L4 20 L5 6" stroke="red"/>
    <path d="M10 11 L10 17 M14 11 L14 17" stroke="red"/>
  </svg>`,

  // 拓扑检查图标 - 橙色圆圈和T字母
  topology: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M2 12 L10 12 M14 12 L22 12" stroke="orange"/>
    <circle cx="12" cy="12" r="8" fill="none" stroke="orange" stroke-dasharray="2,2"/>
    <text x="12" y="16" text-anchor="middle" font-size="8" fill="orange">T</text>
  </svg>`,

  // 范围设置图标 - 紫色虚线框
  range: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <rect x="3" y="3" width="18" height="18" fill="none" stroke="purple" stroke-dasharray="2,2"/>
    <circle cx="12" cy="12" r="2" fill="purple"/>
    <path d="M8 8 L16 16 M16 8 L8 16" stroke="purple" stroke-width="1"/>
  </svg>`,

  // 缩放图标 - 绿色放大镜
  zoom: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="11" cy="11" r="8" fill="none" stroke="green"/>
    <circle cx="11" cy="11" r="3" fill="green"/>
    <line x1="17" y1="17" x2="22" y2="22" stroke="green" stroke-width="3"/>
    <path d="M8 11 L14 11 M11 8 L11 14" stroke="white" stroke-width="2"/>
  </svg>`
};

// ==================== 自定义控件类：绘制工具栏 ====================

/**
 * 自定义绘制工具控件
 * 继承OpenLayers的Control类，创建一个包含绘制工具的工具栏
 *
 * 教学要点：
 * 1. 如何创建自定义控件
 * 2. 控件的事件处理机制
 * 3. SVG图标在控件中的应用
 * 4. CSS样式的应用
 */
class DrawingToolsControl extends Control {
  /**
   * 构造函数
   * @param {Object} opt_options - 配置选项
   */
  constructor(opt_options) {
    const options = opt_options || {};

    // 创建控件容器元素
    const element = document.createElement('div');
    element.className = 'drawing-tools ol-unselectable ol-control';

    // 定义工具按钮配置
    const buttons = [
      { id: 'point', title: '绘制点', svg: svgIcons.point },
      { id: 'line', title: '绘制线', svg: svgIcons.line },
      { id: 'polygon', title: '绘制面', svg: svgIcons.polygon },
      { id: 'clear', title: '清除', svg: svgIcons.clear },
      { id: 'topology', title: '拓扑检查', svg: svgIcons.topology },
      { id: 'range', title: '限制输入范围', svg: svgIcons.range },
      { id: 'zoom', title: '缩放到范围', svg: svgIcons.zoom }
    ];

    // 创建按钮并添加到容器
    buttons.forEach(btn => {
      const button = document.createElement('button');
      button.innerHTML = btn.svg;
      button.title = btn.title;
      button.className = 'tool-button';
      button.addEventListener('click', () => this.handleToolClick(btn.id));
      element.appendChild(button);
    });

    // 调用父类构造函数
    super({
      element: element,
      target: options.target,
    });
  }

  /**
   * 处理工具按钮点击事件
   * @param {string} toolId - 工具ID
   */
  handleToolClick(toolId) {
    switch(toolId) {
      case 'point':
        startDrawing('Point');
        break;
      case 'line':
        startDrawing('LineString');
        break;
      case 'polygon':
        startDrawing('Polygon');
        break;
      case 'clear':
        clearAllDrawings();
        break;
      case 'topology':
        checkTopology();
        break;
      case 'range':
        setDrawingRange();
        break;
      case 'zoom':
        zoomToRange();
        break;
    }
  }
}

// ==================== 图层初始化 ====================

/**
 * 初始化绘制图层
 * 用于显示用户绘制的地理要素
 */
drawSource = new VectorSource();  // 创建矢量数据源
drawLayer = new VectorLayer({
  source: drawSource,
  // 根据几何类型设置不同的样式
  style: function(feature) {
    const geometry = feature.getGeometry();
    if (geometry instanceof Point) {
      // 点样式：红色实心圆圈
      return new Style({
        image: new Circle({
          radius: 6,
          fill: new Fill({ color: 'red' }),
          stroke: new Stroke({ color: 'white', width: 2 })
        })
      });
    } else if (geometry instanceof LineString) {
      // 线样式：蓝色实线
      return new Style({
        stroke: new Stroke({ color: 'blue', width: 3 })
      });
    } else if (geometry instanceof Polygon) {
      // 面样式：绿色半透明填充
      return new Style({
        fill: new Fill({ color: 'rgba(0, 255, 0, 0.3)' }),
        stroke: new Stroke({ color: 'green', width: 2 })
      });
    }
  }
});

/**
 * 初始化高亮图层
 * 用于显示拓扑检查时发现的错误要素
 */
highlightSource = new VectorSource();
highlightLayer = new VectorLayer({
  source: highlightSource,
  style: function(feature) {
    const geometry = feature.getGeometry();
    if (geometry instanceof Point) {
      // 错误点样式：大红色圆圈
      return new Style({
        image: new Circle({
          radius: 8,
          fill: new Fill({ color: 'red' }),
          stroke: new Stroke({ color: 'darkred', width: 3 })
        }),
        zIndex: 1000  // 高层级确保在最上层显示
      });
    } else if (geometry instanceof LineString) {
      // 错误线样式：粗红线
      return new Style({
        stroke: new Stroke({ color: 'red', width: 6 }),
        zIndex: 1000
      });
    } else if (geometry instanceof Polygon) {
      // 错误面样式：红色半透明填充
      return new Style({
        fill: new Fill({ color: 'rgba(255, 0, 0, 0.4)' }),
        stroke: new Stroke({ color: 'red', width: 4 }),
        zIndex: 1000
      });
    }
  }
});

/**
 * 初始化范围图层
 * 用于显示允许的绘制范围（蓝色虚线框）
 */
rangeLayer = new VectorLayer({
  source: new VectorSource(),
  style: new Style({
    fill: new Fill({ color: 'rgba(128, 128, 255, 0.1)' }), // 浅蓝色填充
    stroke: new Stroke({
      color: 'blue',
      width: 2,
      lineDash: [5, 5]  // 虚线样式
    }),
    zIndex: 1  // 低层级，在最底层显示
  })
});

// ==================== 绘制功能核心函数 ====================

/**
 * 开始绘制
 * @param {string} type - 绘制类型（'Point', 'LineString', 'Polygon'）
 *
 * 教学要点：
 * 1. OpenLayers绘制交互的使用
 * 2. 范围限制的实现
 * 3. 坐标系统转换的重要性
 * 4. 用户体验优化
 */
function startDrawing(type) {
  // 移除已存在的绘制交互
  if (drawInteraction) {
    map.removeInteraction(drawInteraction);
  }

  // 检查是否设置了绘制范围
  if (!rangeExtent) {
    app.$notify({
      title: '未设置绘制范围',
      message: '请先点击"限制输入范围"按钮设置绘制范围，然后再开始绘制。',
      type: 'warning',
      duration: 4000,
      position: 'top-right'
    });
    return;
  }

  // 设置当前绘制模式
  drawingMode = type;

  // 创建新的绘制交互
  drawInteraction = new Draw({
    source: drawSource,     // 指定数据源
    type: type              // 指定绘制类型
  });

  /**
   * 绘制完成事件处理
   * 这是范围检查的核心逻辑
   */
  drawInteraction.on('drawend', function(event) {
    const feature = event.feature;

    // 检查绘制内容是否在允许范围内
    if (rangeExtent) {
      const geometry = feature.getGeometry();
      let isWithinRange = true;
      let geometryType = geometry.getType();

      // 根据几何类型进行范围检查
      if (geometryType === 'Point') {
        // 点检查：检查单个坐标点
        const coords = geometry.getCoordinates();
        const lonLat = toLonLat(coords);  // 转换为经纬度
        if (lonLat[0] < rangeExtent[0] || lonLat[0] > rangeExtent[2] ||
            lonLat[1] < rangeExtent[1] || lonLat[1] > rangeExtent[3]) {
          isWithinRange = false;
        }
      } else if (geometryType === 'LineString') {
        // 线检查：检查所有顶点
        const coords = geometry.getCoordinates();
        for (let coord of coords) {
          const lonLat = toLonLat(coord);
          if (lonLat[0] < rangeExtent[0] || lonLat[0] > rangeExtent[2] ||
              lonLat[1] < rangeExtent[1] || lonLat[1] > rangeExtent[3]) {
            isWithinRange = false;
            break;
          }
        }
      } else if (geometryType === 'Polygon') {
        // 面检查：检查所有顶点（包括内环）
        const rings = geometry.getCoordinates();
        for (let ring of rings) {
          for (let coord of ring) {
            const lonLat = toLonLat(coord);
            if (lonLat[0] < rangeExtent[0] || lonLat[0] > rangeExtent[2] ||
                lonLat[1] < rangeExtent[1] || lonLat[1] > rangeExtent[3]) {
              isWithinRange = false;
              break;
            }
          }
          if (!isWithinRange) break;
        }
      }

      // 如果超出范围，删除要素并显示提示
      if (!isWithinRange) {
        // 计算绘制要素的实际范围（用于反馈）
        const extent = geometry.getExtent();
        const minCorner = toLonLat([extent[0], extent[1]]);
        const maxCorner = toLonLat([extent[2], extent[3]]);
        const drawingExtentLonLat = [
          Math.min(minCorner[0], maxCorner[0]),
          Math.min(minCorner[1], maxCorner[1]),
          Math.max(minCorner[0], maxCorner[0]),
          Math.max(minCorner[1], maxCorner[1])
        ];

        // 计算重叠比例
        const intersection = ol.extent.getIntersection(rangeExtent, drawingExtentLonLat);
        const intersectionArea = ol.extent.getArea(intersection);
        const drawingArea = ol.extent.getArea(drawingExtentLonLat);
        const overlapPercentage = drawingArea > 0 ? (intersectionArea / drawingArea * 100).toFixed(1) : 0;

        // 提供详细的错误反馈
        const minX = rangeExtent[0].toFixed(4);
        const minY = rangeExtent[1].toFixed(4);
        const maxX = rangeExtent[2].toFixed(4);
        const maxY = rangeExtent[3].toFixed(4);

        const drawingMinX = drawingExtentLonLat[0].toFixed(4);
        const drawingMinY = drawingExtentLonLat[1].toFixed(4);
        const drawingMaxX = drawingExtentLonLat[2].toFixed(4);
        const drawingMaxY = drawingExtentLonLat[3].toFixed(4);

        // 显示友好的错误提示
        app.$notify({
          title: '绘制超出范围',
          message: `绘制内容超出允许范围！\n\n允许范围：\n经度: ${minX}° 到 ${maxX}°\n纬度: ${minY}° 到 ${maxY}°\n\n绘制范围：\n经度: ${drawingMinX}° 到 ${drawingMaxX}°\n纬度: ${drawingMinY}° 到 ${drawingMaxY}°\n\n重叠比例：${overlapPercentage}%\n\n请重新绘制或调整范围设置。`,
          type: 'error',
          duration: 5000,
          position: 'top-right'
        });

        // 移除超出范围的要素
        drawSource.removeFeature(feature);
        return;
      }
    }
  });

  // 将绘制交互添加到地图
  map.addInteraction(drawInteraction);
}

/**
 * 清除所有绘制内容
 */
function clearAllDrawings() {
  drawSource.clear();
  highlightSource.clear();  // 同时清除高亮显示
  if (drawInteraction) {
    map.removeInteraction(drawInteraction);
    drawInteraction = null;
  }
  drawingMode = null;
}

// ==================== 拓扑检查功能 ====================

/**
 * 拓扑检查函数
 * 使用Turf.js进行空间分析，检查绘制要素的拓扑关系
 *
 * 教学要点：
 * 1. Turf.js空间分析库的使用
 * 2. 各种拓扑关系的检查方法
 * 3. OpenLayers与Turf.js的数据转换
 * 4. 复杂的空间分析逻辑
 */
function checkTopology() {
  const features = drawSource.getFeatures();
  let errors = [];
  let errorFeatureIndices = new Set(); // 记录有错误的要素索引

  // 检查是否有可检查的要素
  if (features.length === 0) {
    app.$message.warning('没有可检查的要素');
    return;
  }

  // 清除之前的高亮显示
  highlightSource.clear();

  // 转换OpenLayers要素为Turf.js格式
  const turfFeatures = features.map((feature, index) => {
    const geometry = feature.getGeometry();
    let turfGeometry;

    // 根据几何类型进行转换
    if (geometry instanceof Point) {
      const coords = toLonLat(geometry.getCoordinates());
      turfGeometry = turf.point(coords);
    } else if (geometry instanceof LineString) {
      const coords = geometry.getCoordinates().map(coord => toLonLat(coord));
      turfGeometry = turf.lineString(coords);
    } else if (geometry instanceof Polygon) {
      const coords = geometry.getCoordinates()[0].map(coord => toLonLat(coord));
      turfGeometry = turf.polygon([coords]);
    }

    // 保留原始信息
    return {
      ...turfGeometry,
      properties: {
        index: index + 1,
        type: geometry.constructor.name,
        originalIndex: index
      }
    };
  });

  // 检查每个要素的拓扑问题
  turfFeatures.forEach((feature, index) => {
    const featureIndex = feature.properties.index;
    const featureType = feature.properties.type;
    const originalIndex = feature.properties.originalIndex;
    let hasError = false;

    // 检查多边形特有的问题
    if (featureType === 'Polygon') {
      try {
        // 检查多边形有效性（不自相交）
        if (!turf.booleanValid(feature)) {
          errors.push(`面${featureIndex}: 多边形自相交`);
          hasError = true;
        }

        // 检查多边形闭合性
        const coords = feature.geometry.coordinates[0];
        if (coords.length < 3) {
          errors.push(`面${featureIndex}: 坐标点数不足，需要至少3个点`);
          hasError = true;
        }

        const firstPoint = coords[0];
        const lastPoint = coords[coords.length - 1];
        if (Math.abs(firstPoint[0] - lastPoint[0]) > 0.000001 ||
            Math.abs(firstPoint[1] - lastPoint[1]) > 0.000001) {
          errors.push(`面${featureIndex}: 多边形未闭合`);
          hasError = true;
        }

        // 检查面积过小
        const area = turf.area(feature);
        if (area < 0.0001) { // 小于约1平方米
          errors.push(`面${featureIndex}: 面积过小`);
          hasError = true;
        }

      } catch (error) {
        errors.push(`面${featureIndex}: 多边形无效 - ${error.message}`);
        hasError = true;
      }
    }

    // 检查线要素特有的问题
    if (featureType === 'LineString') {
      const coords = feature.geometry.coordinates;

      // 检查线段长度
      if (coords.length < 2) {
        errors.push(`线${featureIndex}: 线段至少需要2个点`);
        hasError = true;
      }

      // 检查长度过短
      const length = turf.length(feature, { units: 'meters' });
      if (length < 0.1) {
        errors.push(`线${featureIndex}: 线段长度过短`);
        hasError = true;
      }

      // 检查重复点
      for (let i = 0; i < coords.length - 1; i++) {
        for (let j = i + 1; j < coords.length; j++) {
          if (Math.abs(coords[i][0] - coords[j][0]) < 0.000001 &&
              Math.abs(coords[i][1] - coords[j][1]) < 0.000001) {
            errors.push(`线${featureIndex}: 存在重复坐标点`);
            hasError = true;
            break;
          }
        }
      }
    }

    // 检查点要素的重合问题
    if (featureType === 'Point') {
      turfFeatures.forEach((otherFeature, otherIndex) => {
        if (index !== otherIndex && otherFeature.properties.type === 'Point') {
          const distance = turf.distance(feature, otherFeature, { units: 'meters' });
          if (distance < 0.1) { // 小于10厘米认为重合
            errors.push(`点${featureIndex}与点${otherFeature.properties.index}距离过近（${distance.toFixed(2)}米）`);
            hasError = true;
          }
        }
      });
    }

    // 如果发现错误，记录要素索引
    if (hasError) {
      errorFeatureIndices.add(originalIndex);
    }
  });

  // 检查要素间的拓扑关系
  for (let i = 0; i < turfFeatures.length; i++) {
    for (let j = i + 1; j < turfFeatures.length; j++) {
      const feature1 = turfFeatures[i];
      const feature2 = turfFeatures[j];
      const index1 = feature1.properties.index;
      const index2 = feature2.properties.index;

      // 检查重叠关系
      try {
        if (turf.booleanOverlap(feature1, feature2)) {
          const type1 = feature1.properties.type;
          const type2 = feature2.properties.type;
          errors.push(`${type1}${index1}与${type2}${index2}存在重叠`);
          errorFeatureIndices.add(feature1.properties.originalIndex);
          errorFeatureIndices.add(feature2.properties.originalIndex);
        }
      } catch (error) {
        // 重叠检查可能失败，跳过
      }

      // 检查交叉关系
      if (feature1.properties.type === 'LineString' && feature2.properties.type === 'LineString') {
        try {
          if (turf.booleanCrosses(feature1, feature2)) {
            errors.push(`线${index1}与线${index2}存在交叉`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 交叉检查可能失败
        }
      }

      // 多边形间的复杂拓扑检查
      if (feature1.properties.type === 'Polygon' && feature2.properties.type === 'Polygon') {
        // 检查相交
        if (turf.booleanIntersects(feature1, feature2)) {
          try {
            const intersection = turf.intersect(feature1, feature2);
            if (intersection) {
              const intersectionArea = turf.area(intersection);
              if (intersectionArea > 0.0001) { // 大于1平方米的相交
                errors.push(`面${index1}与面${index2}相交，相交面积约${intersectionArea.toFixed(2)}平方米`);
                errorFeatureIndices.add(feature1.properties.originalIndex);
                errorFeatureIndices.add(feature2.properties.originalIndex);
              }
            }
          } catch (error) {
            errors.push(`面${index1}与面${index2}存在相交，但计算相交面积时出错`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        }

        // 检查包含关系
        try {
          if (turf.booleanContains(feature1, feature2)) {
            const area1 = turf.area(feature1);
            const area2 = turf.area(feature2);
            errors.push(`面${index1}包含面${index2}（包含${(area2/area1*100).toFixed(1)}%的面积）`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 包含关系检查可能失败
        }
      }
    }
  }

  // 创建高亮要素显示错误
  errorFeatureIndices.forEach(index => {
    const originalFeature = features[index];
    if (originalFeature) {
      const geometry = originalFeature.getGeometry().clone();
      const highlightFeature = new Feature({
        geometry: geometry
      });
      highlightSource.addFeature(highlightFeature);
    }
  });

  // 显示检查结果
  if (errors.length > 0) {
    const errorMessage = `发现 ${errors.length} 个拓扑错误：\n\n${errors.join('\n')}\n\n有错误的要素已在地图上高亮显示（红色加粗）`;
    app.$alert(errorMessage, '拓扑检查结果', {
      confirmButtonText: '确定',
      type: 'error',
      customClass: 'topology-error-dialog'
    });
  } else {
    app.$message.success('拓扑检查完成，未发现错误');
  }
}

// ==================== 范围管理功能 ====================

/**
 * 打开范围设置对话框
 */
function setDrawingRange() {
  app.dialogVisible = true;
}

/**
 * 缩放到设置的范围
 * 优先使用范围图层，如果没有则使用经纬度范围
 */
function zoomToRange() {
  // 优先缩放到范围图层
  if (rangeLayer.getSource().getFeatures().length > 0) {
    map.getView().fit(rangeLayer.getSource().getExtent(), {
      padding: [50, 50, 50, 50],  // 内边距
      duration: 1000  // 动画持续时间
    });
  } else if (rangeExtent) {
    // 如果没有范围图层但有范围设置，使用经纬度范围
    const minCoord = fromLonLat([rangeExtent[0], rangeExtent[1]]);
    const maxCoord = fromLonLat([rangeExtent[2], rangeExtent[3]]);
    const projectedExtent = [
      Math.min(minCoord[0], maxCoord[0]),
      Math.min(minCoord[1], maxCoord[1]),
      Math.max(minCoord[0], maxCoord[0]),
      Math.max(minCoord[1], maxCoord[1])
    ];

    map.getView().fit(projectedExtent, {
      padding: [50, 50, 50, 50],
      duration: 1000
    });
  } else {
    app.$notify({
      title: '未设置范围',
      message: '请先设置绘制范围，然后再使用缩放功能。',
      type: 'warning',
      duration: 3000,
      position: 'top-right'
    });
  }
}

// ==================== Vue.js 应用实例 ====================

/**
 * 创建Vue应用，处理用户界面交互
 *
 * 教学要点：
 * 1. Vue.js与地图组件的集成
 * 2. 表单验证的实现
 * 3. 用户界面的响应式设计
 * 4. 错误处理和用户反馈
 */
const app = new Vue({
  el: '#app',
  data() {
    return {
      dialogVisible: false,  // 对话框显示状态
      rangeForm: {          // 范围表单数据
        minX: '',
        minY: '',
        maxX: '',
        maxY: ''
      },
      rules: {              // 表单验证规则
        minX: [
          { required: true, message: '请输入最小经度', trigger: 'blur' },
          {
            validator: (rule, value, callback) => {
              if (!value && value !== 0) {
                callback(new Error('请输入最小经度'));
              } else if (isNaN(value)) {
                callback(new Error('请输入有效的数字'));
              } else if (parseFloat(value) < -180 || parseFloat(value) > 180) {
                callback(new Error('经度范围: -180 到 180'));
              } else {
                callback();
              }
            },
            trigger: 'blur'
          }
        ],
        minY: [
          { required: true, message: '请输入最小纬度', trigger: 'blur' },
          {
            validator: (rule, value, callback) => {
              if (!value && value !== 0) {
                callback(new Error('请输入最小纬度'));
              } else if (isNaN(value)) {
                callback(new Error('请输入有效的数字'));
              } else if (parseFloat(value) < -90 || parseFloat(value) > 90) {
                callback(new Error('纬度范围: -90 到 90'));
              } else {
                callback();
              }
            },
            trigger: 'blur'
          }
        ],
        maxX: [
          { required: true, message: '请输入最大经度', trigger: 'blur' },
          {
            validator: (rule, value, callback) => {
              if (!value && value !== 0) {
                callback(new Error('请输入最大经度'));
              } else if (isNaN(value)) {
                callback(new Error('请输入有效的数字'));
              } else if (parseFloat(value) < -180 || parseFloat(value) > 180) {
                callback(new Error('经度范围: -180 到 180'));
              } else {
                callback();
              }
            },
            trigger: 'blur'
          }
        ],
        maxY: [
          { required: true, message: '请输入最大纬度', trigger: 'blur' },
          {
            validator: (rule, value, callback) => {
              if (!value && value !== 0) {
                callback(new Error('请输入最大纬度'));
              } else if (isNaN(value)) {
                callback(new Error('请输入有效的数字'));
              } else if (parseFloat(value) < -90 || parseFloat(value) > 90) {
                callback(new Error('纬度范围: -90 到 90'));
              } else {
                callback();
              }
            },
            trigger: 'blur'
          }
        ]
      }
    }
  },
  methods: {
    /**
     * 关闭对话框
     */
    handleClose() {
      this.dialogVisible = false;
    },

    /**
     * 验证数字输入
     * @param {string} field - 字段名
     */
    validateNumber(field) {
      const value = this.rangeForm[field];
      if (value === '' || value === null || value === undefined) {
        return;
      }

      // 尝试转换为数字
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        this.rangeForm[field] = '';
        this.$message.error(`请输入有效的数字`);
        return;
      }

      // 根据字段类型进行范围验证
      if (field.includes('X')) { // 经度
        if (numValue < -180 || numValue > 180) {
          this.rangeForm[field] = '';
          this.$message.error('经度范围必须在 -180 到 180 之间');
        }
      } else if (field.includes('Y')) { // 纬度
        if (numValue < -90 || numValue > 90) {
          this.rangeForm[field] = '';
          this.$message.error('纬度范围必须在 -90 到 90 之间');
        }
      }
    },

    /**
     * 确认设置范围
     */
    confirmRange() {
      this.$refs.rangeForm.validate((valid) => {
        if (valid) {
          const minX = parseFloat(this.rangeForm.minX);
          const minY = parseFloat(this.rangeForm.minY);
          const maxX = parseFloat(this.rangeForm.maxX);
          const maxY = parseFloat(this.rangeForm.maxY);

          // 检查是否为有效数字
          if (isNaN(minX) || isNaN(minY) || isNaN(maxX) || isNaN(maxY)) {
            this.$message.error('请输入有效的数字');
            return;
          }

          // 检查经纬度范围
          if (minX < -180 || minX > 180 || maxX < -180 || maxX > 180) {
            this.$message.error('经度范围必须在 -180 到 180 之间');
            return;
          }

          if (minY < -90 || minY > 90 || maxY < -90 || maxY > 90) {
            this.$message.error('纬度范围必须在 -90 到 90 之间');
            return;
          }

          // 检查最小值小于最大值
          if (minX >= maxX || minY >= maxY) {
            this.$message.error('最小值必须小于最大值');
            return;
          }

          // 检查范围大小是否合理
          const xRange = maxX - minX;
          const yRange = maxY - minY;
          if (xRange < 0.001 || yRange < 0.001) {
            this.$message.error('范围过小，请设置更大的绘制区域');
            return;
          }

          // 保存范围设置
          rangeExtent = [minX, minY, maxX, maxY];

          // 转换坐标为地图投影格式用于显示
          const minCoord = fromLonLat([minX, minY]);
          const maxCoord = fromLonLat([maxX, maxY]);

          // 创建范围显示要素
          const rangeFeature = new Feature({
            geometry: new Polygon([[
              [minCoord[0], minCoord[1]],
              [maxCoord[0], minCoord[1]],
              [maxCoord[0], maxCoord[1]],
              [minCoord[0], maxCoord[1]],
              [minCoord[0], minCoord[1]]
            ]])
          });

          // 更新范围图层显示
          rangeLayer.getSource().clear();
          rangeLayer.getSource().addFeature(rangeFeature);

          // 缩放到设置的范围
          if (rangeLayer.getSource().getFeatures().length > 0) {
            map.getView().fit(rangeLayer.getSource().getExtent(), {
              padding: [50, 50, 50, 50],
              duration: 1000
            });
          } else {
            // 后备方案：转换经纬度范围为投影坐标
            const minCoord = fromLonLat([rangeExtent[0], rangeExtent[1]]);
            const maxCoord = fromLonLat([rangeExtent[2], rangeExtent[3]]);
            const projectedExtent = [
              Math.min(minCoord[0], maxCoord[0]),
              Math.min(minCoord[1], maxCoord[1]),
              Math.max(minCoord[0], maxCoord[0]),
              Math.max(minCoord[1], maxCoord[1])
            ];

            map.getView().fit(projectedExtent, {
              padding: [50, 50, 50, 50],
              duration: 1000
            });
          }

          // 显示成功通知
          this.$notify({
            title: '范围设置成功',
            message: `绘制范围已设置为：\n经度: ${minX}° 到 ${maxX}°\n纬度: ${minY}° 到 ${maxY}°\n\n蓝色虚线框显示允许的绘制范围。`,
            type: 'success',
            duration: 4000,
            position: 'top-right'
          });

          this.dialogVisible = false;

          // 清空表单
          this.rangeForm = {
            minX: '',
            minY: '',
            maxX: '',
            maxY: ''
          };
        } else {
          return false;
        }
      });
    }
  }
});

// ==================== 地图初始化 ====================

/**
 * 创建OpenLayers地图实例
 *
 * 教学要点：
 * 1. 地图配置和初始化
 * 2. 图层组织和管理
 * 3. 坐标系统和投影
 * 4. 控件的使用
 */
const map = new Map({
  // 配置控件（包含默认控件和自定义工具栏）
  controls: defaultControls().extend([new DrawingToolsControl()]),

  // 配置图层（按显示顺序）
  layers: [
    // 底图图层：天地图矢量瓦片
    new TileLayer({
      source: new XYZ({
        url: 'http://t{0-7}.tianditu.gov.cn/vec_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=vec&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=' + tk,
        projection: 'EPSG:3857'  // Web墨卡托投影
      })
    }),

    // 注记图层：天地图注记瓦片
    new TileLayer({
      source: new XYZ({
        url: 'http://t{0-7}.tianditu.gov.cn/cva_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=cva&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=' + tk,
        projection: 'EPSG:3857'
      })
    }),

    // 范围图层：显示允许的绘制范围
    rangeLayer,

    // 绘制图层：显示用户绘制的要素
    drawLayer,

    // 高亮图层：显示拓扑错误
    highlightLayer
  ],

  // 地图容器
  target: 'map',

  // 地图视图配置
  view: new View({
    center: fromLonLat([108.9647, 34.2683]), // 西安中心坐标（转换为投影坐标）
    zoom: 10,                               // 缩放级别
    rotation: 0                             // 旋转角度
  }),
});

/**
 * 文件说明
 * ===========
 *
 * 本文件是一个完整的OpenLayers地图绘制工具教学案例，展示了：
 *
 * 1. **OpenLayers基础架构**：地图、图层、视图、控件等核心概念
 * 2. **绘制功能**：点、线、面的绘制交互实现
 * 3. **坐标系统**：经纬度与投影坐标的转换
 * 4. **空间分析**：使用Turf.js进行拓扑检查
 * 5. **用户界面**：Vue.js与地图组件的集成
 * 6. **用户体验**：表单验证、错误处理、用户反馈
 *
 * 技术栈：
 * - OpenLayers 6.x
 * - Vue.js 2.x
 * - Element UI
 * - Turf.js
 * - 天地图API
 *
 * 适合人群：
 * - WebGIS开发初学者
 * - 想要学习OpenLayers的开发者
 * - 需要地图绘制功能的项目
 *
 * 使用方法：
 * 1. 安装依赖：npm install ol vue element-ui @turf/turf
 * 2. 配置天地图API密钥
 * 3. 运行项目
 *
 * 扩展建议：
 * 1. 添加更多绘制工具（矩形、圆形等）
 * 2. 实现要素编辑功能
 * 3. 添加数据导入导出功能
 * 4. 集成其他地图服务
 * 5. 优化性能和用户体验
 */