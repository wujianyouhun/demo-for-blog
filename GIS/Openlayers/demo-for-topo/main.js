import Map from 'ol/Map.js';
import View from 'ol/View.js';
import Control from 'ol/control/Control.js';
import {defaults as defaultControls} from 'ol/control/defaults.js';
import TileLayer from 'ol/layer/Tile.js';
import XYZ from 'ol/source/XYZ.js';
import VectorLayer from 'ol/layer/Vector.js';
import VectorSource from 'ol/source/Vector.js';
import Draw from 'ol/interaction/Draw.js';
import {fromLonLat, toLonLat} from 'ol/proj.js';
import Feature from 'ol/Feature.js';
import Point from 'ol/geom/Point.js';
import LineString from 'ol/geom/LineString.js';
import Polygon from 'ol/geom/Polygon.js';
import {Style, Circle, Fill, Stroke} from 'ol/style.js';
import {Icon} from 'ol/style.js';
import * as olExtent from 'ol/extent.js';
import * as turf from '@turf/turf';

// Define drawing tools control
const tk = 'f4d0553a23372a2f48c74851c7e46f4d';
let drawInteraction;
let drawLayer;
let drawingMode = null;
let drawSource;
let rangeExtent = null;

// SVG icon definitions
const svgIcons = {
  point: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="12" cy="12" r="10" fill="red"/>
    <circle cx="12" cy="12" r="3" fill="white"/>
  </svg>`,
  line: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M4 12 L20 12" stroke="blue"/>
    <circle cx="4" cy="12" r="3" fill="blue"/>
    <circle cx="20" cy="12" r="3" fill="blue"/>
  </svg>`,
  polygon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M12 2 L22 8 L18 20 L6 20 L2 8 Z" fill="green" fill-opacity="0.3" stroke="green"/>
  </svg>`,
  clear: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M3 6 L21 6 M19 6 L20 20 L4 20 L5 6" stroke="red"/>
    <path d="M10 11 L10 17 M14 11 L14 17" stroke="red"/>
  </svg>`,
  topology: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M2 12 L10 12 M14 12 L22 12" stroke="orange"/>
    <circle cx="12" cy="12" r="8" fill="none" stroke="orange" stroke-dasharray="2,2"/>
    <text x="12" y="16" text-anchor="middle" font-size="8" fill="orange">T</text>
  </svg>`,
  range: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <rect x="3" y="3" width="18" height="18" fill="none" stroke="purple" stroke-dasharray="2,2"/>
    <circle cx="12" cy="12" r="2" fill="purple"/>
    <path d="M8 8 L16 16 M16 8 L8 16" stroke="purple" stroke-width="1"/>
  </svg>`
};

class DrawingToolsControl extends Control {
  constructor(opt_options) {
    const options = opt_options || {};

    const element = document.createElement('div');
    element.className = 'drawing-tools ol-unselectable ol-control';

    const buttons = [
      { id: 'point', title: '绘制点', svg: svgIcons.point },
      { id: 'line', title: '绘制线', svg: svgIcons.line },
      { id: 'polygon', title: '绘制面', svg: svgIcons.polygon },
      { id: 'clear', title: '清除', svg: svgIcons.clear },
      { id: 'topology', title: '拓扑检查', svg: svgIcons.topology },
      { id: 'range', title: '限制输入范围', svg: svgIcons.range }
    ];

    buttons.forEach(btn => {
      const button = document.createElement('button');
      button.innerHTML = btn.svg;
      button.title = btn.title;
      button.className = 'tool-button';
      button.addEventListener('click', () => this.handleToolClick(btn.id));
      element.appendChild(button);
    });

    super({
      element: element,
      target: options.target,
    });
  }

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
    }
  }
}

// Initialize drawing layer
drawSource = new VectorSource();
drawLayer = new VectorLayer({
  source: drawSource,
  style: function(feature) {
    const geometry = feature.getGeometry();
    if (geometry instanceof Point) {
      return new Style({
        image: new Circle({
          radius: 6,
          fill: new Fill({ color: 'red' }),
          stroke: new Stroke({ color: 'white', width: 2 })
        })
      });
    } else if (geometry instanceof LineString) {
      return new Style({
        stroke: new Stroke({ color: 'blue', width: 3 })
      });
    } else if (geometry instanceof Polygon) {
      return new Style({
        fill: new Fill({ color: 'rgba(0, 255, 0, 0.3)' }),
        stroke: new Stroke({ color: 'green', width: 2 })
      });
    }
  }
});

// Initialize highlight layer for topology errors
highlightSource = new VectorSource();
highlightLayer = new VectorLayer({
  source: highlightSource,
  style: function(feature) {
    const geometry = feature.getGeometry();
    if (geometry instanceof Point) {
      return new Style({
        image: new Circle({
          radius: 8,
          fill: new Fill({ color: 'red' }),
          stroke: new Stroke({ color: 'darkred', width: 3 })
        }),
        zIndex: 1000
      });
    } else if (geometry instanceof LineString) {
      return new Style({
        stroke: new Stroke({ color: 'red', width: 6 }),
        zIndex: 1000
      });
    } else if (geometry instanceof Polygon) {
      return new Style({
        fill: new Fill({ color: 'rgba(255, 0, 0, 0.4)' }),
        stroke: new Stroke({ color: 'red', width: 4 }),
        zIndex: 1000
      });
    }
  }
});

// Drawing functions
function startDrawing(type) {
  if (drawInteraction) {
    map.removeInteraction(drawInteraction);
  }

  drawingMode = type;
  drawInteraction = new Draw({
    source: drawSource,
    type: type
  });

  drawInteraction.on('drawend', function(event) {
    const feature = event.feature;

    // Check if drawing is within range
    if (rangeExtent) {
      const geometry = feature.getGeometry();
      const extent = geometry.getExtent();

      if (!ol.extent.containsExtent(rangeExtent, extent)) {
        alert('绘制内容超出允许范围！');
        drawSource.removeFeature(feature);
        return;
      }
    }
  });

  map.addInteraction(drawInteraction);
}

function clearAllDrawings() {
  drawSource.clear();
  highlightSource.clear(); // 同时清除高亮
  if (drawInteraction) {
    map.removeInteraction(drawInteraction);
    drawInteraction = null;
  }
  drawingMode = null;
}

function checkTopology() {
  const features = drawSource.getFeatures();
  let errors = [];
  let errorFeatureIndices = new Set(); // 记录有错误的要素索引

  if (features.length === 0) {
    app.$message.warning('没有可检查的要素');
    return;
  }

  // 清除之前的高亮
  highlightSource.clear();

  // 转换OpenLayers要素为GeoJSON格式以便使用Turf.js
  const turfFeatures = features.map((feature, index) => {
    const geometry = feature.getGeometry();
    let turfGeometry;

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

    return {
      ...turfGeometry,
      properties: { index: index + 1, type: geometry.constructor.name, originalIndex: index }
    };
  });

  // 检查每个要素的拓扑问题
  turfFeatures.forEach((feature, index) => {
    const featureIndex = feature.properties.index;
    const featureType = feature.properties.type;
    const originalIndex = feature.properties.originalIndex;
    let hasError = false;

    // 检查多边形自相交
    if (featureType === 'Polygon') {
      try {
        // 检查多边形是否有效（不自相交）
        if (!turf.booleanValid(feature)) {
          errors.push(`面${featureIndex}: 多边形自相交`);
          hasError = true;
        }

        // 检查多边形是否闭合
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

    // 检查线要素
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

      // 检查是否有重复点
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

    // 检查点要素
    if (featureType === 'Point') {
      // 检查与其他点的重合
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

    // 如果有错误，添加到高亮集合
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

      // 检查重叠（仅在适当几何类型间）
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

      // 检查交叉（仅在适当几何类型间）
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
      } else if (feature1.properties.type === 'LineString' && feature2.properties.type === 'Polygon') {
        try {
          if (turf.booleanCrosses(feature1, feature2)) {
            errors.push(`线${index1}与面${index2}存在交叉`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 交叉检查可能失败
        }
      } else if (feature1.properties.type === 'Polygon' && feature2.properties.type === 'LineString') {
        try {
          if (turf.booleanCrosses(feature2, feature1)) {
            errors.push(`线${index2}与面${index1}存在交叉`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 交叉检查可能失败
        }
      }
      // 多边形间不使用booleanCrosses，因为它不支持多边形

      // 多边形间的高级拓扑检查
      if (feature1.properties.type === 'Polygon' && feature2.properties.type === 'Polygon') {
        // 检查多边形相交
        if (turf.booleanIntersects(feature1, feature2)) {
          // 计算相交面积
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
          if (turf.booleanContains(feature2, feature1)) {
            const area1 = turf.area(feature1);
            const area2 = turf.area(feature2);
            errors.push(`面${index2}包含面${index1}（包含${(area1/area2*100).toFixed(1)}%的面积）`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 包含关系检查可能失败
        }

        // 检查相邻关系（共享边界）
        try {
          const sharedDistance = turf.distanceToLine(feature1, feature2, { units: 'meters' });
          if (sharedDistance < 0.1) { // 距离小于10厘米认为相邻
            errors.push(`面${index1}与面${index2}相邻`);
            // 相邻不是错误，所以不添加到高亮集合
          }
        } catch (error) {
          // 如果无法计算距离，跳过相邻检查
        }

        // 检查多边形间的缝隙
        try {
          const union = turf.union(feature1, feature2);
          if (union) {
            const unionArea = turf.area(union);
            const area1 = turf.area(feature1);
            const area2 = turf.area(feature2);
            const gapArea = unionArea - area1 - area2;

            if (gapArea > 0.0001) { // 缝隙大于1平方米
              errors.push(`面${index1}与面${index2}之间存在缝隙，面积约${gapArea.toFixed(2)}平方米`);
              errorFeatureIndices.add(feature1.properties.originalIndex);
              errorFeatureIndices.add(feature2.properties.originalIndex);
            }
          }
        } catch (error) {
          // 联合操作可能失败，跳过缝隙检查
        }
      }

      // 线与多边形的拓扑检查
      if (feature1.properties.type === 'LineString' && feature2.properties.type === 'Polygon') {
        // 检查线是否在多边形内
        try {
          if (turf.booleanContains(feature2, feature1)) {
            errors.push(`线${index1}完全在面${index2}内`);
            // 线在面内不是错误，所以不添加到高亮集合
          }
        } catch (error) {
          // 包含检查可能失败
        }

        // 检查线与多边形边界的交点
        try {
          const intersectionPoints = turf.lineIntersect(feature1, feature2);
          if (intersectionPoints.features.length > 0) {
            errors.push(`线${index1}与面${index2}边界有${intersectionPoints.features.length}个交点`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 交点计算可能失败
        }
      }

      // 多边形与线的拓扑检查
      if (feature1.properties.type === 'Polygon' && feature2.properties.type === 'LineString') {
        // 检查线是否在多边形内
        try {
          if (turf.booleanContains(feature1, feature2)) {
            errors.push(`线${index2}完全在面${index1}内`);
            // 线在面内不是错误，所以不添加到高亮集合
          }
        } catch (error) {
          // 包含检查可能失败
        }

        // 检查线与多边形边界的交点
        try {
          const intersectionPoints = turf.lineIntersect(feature2, feature1);
          if (intersectionPoints.features.length > 0) {
            errors.push(`线${index2}与面${index1}边界有${intersectionPoints.features.length}个交点`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } catch (error) {
          // 交点计算可能失败
        }
      }

      // 检查线段的端点是否连接（悬挂线检查）
      if (feature1.properties.type === 'LineString' && feature2.properties.type === 'LineString') {
        const coords1 = feature1.geometry.coordinates;
        const coords2 = feature2.geometry.coordinates;

        const endpoints1 = [coords1[0], coords1[coords1.length - 1]];
        const endpoints2 = [coords2[0], coords2[coords2.length - 1]];

        let hasConnection = false;
        let connectionPoints = [];

        for (let ep1 of endpoints1) {
          for (let ep2 of endpoints2) {
            const distance = turf.distance(turf.point(ep1), turf.point(ep2), { units: 'meters' });
            if (distance < 0.1) {
              hasConnection = true;
              connectionPoints.push({ point: ep1, distance: distance });
            }
          }
        }

        if (!hasConnection) {
          // 检查是否应该连接（距离较近的端点）
          let minDistance = Infinity;
          for (let ep1 of endpoints1) {
            for (let ep2 of endpoints2) {
              const distance = turf.distance(turf.point(ep1), turf.point(ep2), { units: 'meters' });
              minDistance = Math.min(minDistance, distance);
            }
          }

          if (minDistance < 1.0) { // 小于1米认为是潜在的悬挂线
            errors.push(`线${index1}与线${index2}存在潜在悬挂点（距离${minDistance.toFixed(2)}米）`);
            errorFeatureIndices.add(feature1.properties.originalIndex);
            errorFeatureIndices.add(feature2.properties.originalIndex);
          }
        } else if (connectionPoints.length === 1) {
          // 只有一个端点连接，可能是悬挂线
          errors.push(`线${index1}与线${index2}只有一个端点连接，可能存在悬挂`);
          errorFeatureIndices.add(feature1.properties.originalIndex);
          errorFeatureIndices.add(feature2.properties.originalIndex);
        }
      }
    }
  }

  // 多边形组的高级拓扑检查
  const polygonFeatures = turfFeatures.filter(f => f.properties.type === 'Polygon');
  if (polygonFeatures.length > 2) {
    // 检查多边形组的拓扑一致性
    try {
      // 检查是否有重叠区域
      for (let i = 0; i < polygonFeatures.length; i++) {
        for (let j = i + 1; j < polygonFeatures.length; j++) {
          const poly1 = polygonFeatures[i];
          const poly2 = polygonFeatures[j];

          // 检查边界一致性
          try {
            const sharedCoords = [];
            const coords1 = poly1.geometry.coordinates[0];
            const coords2 = poly2.geometry.coordinates[0];

            for (let coord1 of coords1) {
              for (let coord2 of coords2) {
                const distance = turf.distance(turf.point(coord1), turf.point(coord2), { units: 'meters' });
                if (distance < 0.1) {
                  sharedCoords.push({ coord: coord1, distance: distance });
                }
              }
            }

            if (sharedCoords.length >= 2) {
              errors.push(`面${poly1.properties.index}与面${poly2.properties.index}共享${sharedCoords.length}个边界点`);
            }
          } catch (error) {
            // 边界检查可能失败
          }
        }
      }
    } catch (error) {
      // 多边形组检查可能失败
    }
  }

  // 创建高亮要素
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

  // 显示结果
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

function setDrawingRange() {
  // 打开Vue对话框
  app.dialogVisible = true;
}

// 创建Vue应用
const app = new Vue({
  el: '#app',
  data() {
    return {
      dialogVisible: false,
      rangeForm: {
        minX: '',
        minY: '',
        maxX: '',
        maxY: ''
      },
      rules: {
        minX: [{ required: true, message: '请输入最小X坐标', trigger: 'blur' }],
        minY: [{ required: true, message: '请输入最小Y坐标', trigger: 'blur' }],
        maxX: [{ required: true, message: '请输入最大X坐标', trigger: 'blur' }],
        maxY: [{ required: true, message: '请输入最大Y坐标', trigger: 'blur' }]
      }
    }
  },
  methods: {
    handleClose() {
      this.dialogVisible = false;
    },
    confirmRange() {
      this.$refs.rangeForm.validate((valid) => {
        if (valid) {
          const minX = parseFloat(this.rangeForm.minX);
          const minY = parseFloat(this.rangeForm.minY);
          const maxX = parseFloat(this.rangeForm.maxX);
          const maxY = parseFloat(this.rangeForm.maxY);

          if (minX >= maxX || minY >= maxY) {
            this.$message.error('最小值必须小于最大值');
            return;
          }

          rangeExtent = [minX, minY, maxX, maxY];
          this.$message.success(`绘制范围已设置为：\nX: ${minX} 到 ${maxX}\nY: ${minY} 到 ${maxY}`);
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

const map = new Map({
  controls: defaultControls().extend([new DrawingToolsControl()]),
  layers: [
    new TileLayer({
      source: new XYZ({
        url: 'http://t{0-7}.tianditu.gov.cn/vec_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=vec&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=' + tk,
        projection: 'EPSG:3857'
      })
    }),
    new TileLayer({
      source: new XYZ({
        url: 'http://t{0-7}.tianditu.gov.cn/cva_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=cva&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=' + tk,
        projection: 'EPSG:3857'
      })
    }),
    drawLayer,
    highlightLayer
  ],
  target: 'map',
  view: new View({
    center: fromLonLat([108.9647, 34.2683]), // 西安坐标
    zoom: 10,
    rotation: 0
  }),
});
