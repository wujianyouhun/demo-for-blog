/**
 * OpenLayers 轨迹动画教学案例
 *
 * 本案例演示了如何使用 OpenLayers 实现以下功能：
 * 1. 加载和显示天地图瓦片图层
 * 2. 从 GeoJSON 文件加载轨迹数据
 * 3. 实现轨迹动画效果
 * 4. 添加交互式控制面板
 * 5. 自定义地图标记和样式
 */

// 导入 OpenLayers 核心模块
import Feature from 'ol/Feature.js';          // 地理要素类
import Map from 'ol/Map.js';                  // 地图主类
import View from 'ol/View.js';                // 地图视图类
import GeoJSON from 'ol/format/GeoJSON.js';   // GeoJSON 数据格式解析器
import Point from 'ol/geom/Point.js';         // 点几何对象
import LineString from 'ol/geom/LineString.js'; // 线几何对象
import TileLayer from 'ol/layer/Tile.js';     // 瓦片图层类
import VectorLayer from 'ol/layer/Vector.js'; // 矢量图层类
import {getVectorContext} from 'ol/render.js'; // 获取矢量渲染上下文
import XYZ from 'ol/source/XYZ.js';           // XYZ 瓦片数据源
import VectorSource from 'ol/source/Vector.js'; // 矢量数据源
import CircleStyle from 'ol/style/Circle.js'; // 圆形样式
import Fill from 'ol/style/Fill.js';           // 填充样式
import Icon from 'ol/style/Icon.js';           // 图标样式
import Stroke from 'ol/style/Stroke.js';       // 描边样式
import Style from 'ol/style/Style.js';         // 样式类
import {fromLonLat} from 'ol/proj.js';         // 坐标投影转换工具

// =============================================================================
// 1. 配置天地图服务
// =============================================================================

// 天地图 API 密钥 - 用于访问天地图瓦片服务
// 注意：实际项目中请使用您自己的密钥
const tiandituKey = 'f4d0553a23372a2f48c74851c7e46f4d';

// 创建天地图矢量图层（底图）
// vec_w 表示经纬度坐标系的矢量地图
const vecLayer = new TileLayer({
  source: new XYZ({
    // 天地图 WMTS 服务 URL 模板
    // {0-7} 表示子域，用于分散请求负载
    // {z} 缩放级别，{x} 列号，{y} 行号
    // tk 参数是 API 密钥
    url: 'https://t{0-7}.tianditu.gov.cn/vec_w/wmts?layer=vec&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=' + tiandituKey,
  })
});

// 创建天地图注记图层（道路、地名等标注信息）
// cva_w 表示经纬度坐标系的中文注记
const annoLayer = new TileLayer({
  source: new XYZ({
    url: 'https://t{0-7}.tianditu.gov.cn/cva_w/wmts?layer=cva&style=default&tilematrixset=w&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=' + tiandituKey,
  })
});

// =============================================================================
// 2. 初始化地图
// =============================================================================

// 设置地图初始中心点为西安市中心
// fromLonLat() 将经纬度坐标转换为 Web Mercator 投影坐标（EPSG:3857）
let center = fromLonLat([108.940174, 34.341568]); // 西安坐标

// 创建地图实例
const map = new Map({
  // 指定地图容器的 DOM 元素
  target: document.getElementById('map'),

  // 配置地图视图
  view: new View({
    center: center,           // 地图中心点
    zoom: 15,                 // 初始缩放级别（15 显示西安市街景）
    minZoom: 12,              // 最小缩放级别（限制缩放范围）
    maxZoom: 19,              // 最大缩放级别
  }),

  // 添加图层（按顺序叠加）
  layers: [
    vecLayer,     // 底层：矢量地图
    annoLayer     // 顶层：注记信息
  ],
});

// =============================================================================
// 3. 加载和显示轨迹数据
// =============================================================================

// 异步加载轨迹数据（GeoJSON 格式）
fetch('tracke.geojson').then(function (response) {
  response.json().then(function (geojson) {

    // 创建 GeoJSON 格式解析器
    const format = new GeoJSON();

    // 将 GeoJSON 数据解析为 OpenLayers 要素数组
    // featureProjection 指定目标坐标系为 Web Mercator
    const features = format.readFeatures(geojson, {
      featureProjection: 'EPSG:3857'
    });

    // 获取第一个要素的几何对象（MultiLineString 类型）
    const multiLineString = features[0].getGeometry();

    // 提取第一个 LineString 的坐标数组
    // 因为是闭合轨迹，所以只有一个 LineString
    const coordinates = multiLineString.getCoordinates()[0];

    // 创建 LineString 几何对象，用于动画
    const route = new LineString(coordinates);

    // =============================================================================
    // 4. 自动调整地图视图以适应轨迹
    // =============================================================================

    // 计算轨迹的边界范围
    const extent = route.getExtent();

    // 计算轨迹的中心点
    center = [
      (extent[0] + extent[2]) / 2,  // 中心点 X 坐标
      (extent[1] + extent[3]) / 2   // 中心点 Y 坐标
    ];

    // 更新地图视图中心点到轨迹中心
    map.getView().setCenter(center);

    // 自动计算合适的缩放级别以完整显示轨迹
    const width = extent[2] - extent[0];     // 轨迹宽度
    const height = extent[3] - extent[1];    // 轨迹高度
    const maxDimension = Math.max(width, height);  // 取最大尺寸

    // 根据轨迹尺寸计算缩放级别
    // 使用对数函数计算，确保轨迹在视口中完整显示
    const zoom = Math.max(2, Math.min(19, Math.log(1000 / maxDimension) / Math.log(2)));
    map.getView().setZoom(zoom);

    // =============================================================================
    // 5. 创建地图要素
    // =============================================================================

    // 创建轨迹线要素
    const routeFeature = new Feature({
      type: 'route',           // 要素类型标识
      geometry: route,         // 几何对象
    });

    // 获取轨迹起点坐标
    const startPoint = route.getFirstCoordinate();

    // 创建起点标记要素
    const startMarker = new Feature({
      type: 'icon',                        // 要素类型
      geometry: new Point(startPoint),     // 起点点位
    });

    // 创建终点标记要素（闭合轨迹，终点与起点相同）
    const endMarker = new Feature({
      type: 'icon',
      geometry: new Point(startPoint),
    });

    // 创建固定的起点标记（使用 SVG 图标）
    const fixedStartMarker = new Feature({
      type: 'startMarker',
      geometry: new Point(startPoint),
    });

    // 创建动画标记要素（用于轨迹动画）
    const position = startMarker.getGeometry().clone();
    const geoMarker = new Feature({
      type: 'geoMarker',
      geometry: position,
    });

    // =============================================================================
    // 6. 定义要素样式
    // =============================================================================

    const styles = {
      // 轨迹线样式
      'route': new Style({
        stroke: new Stroke({
          width: 6,                           // 线条宽度
          color: [0, 139, 139, 0.8],        // 青绿色，带透明度
        }),
      }),

      // 普通标记样式
      'icon': new Style({
        image: new Icon({
          anchor: [0.5, 1],                 // 锚点位置（底部中心）
          src: 'data/icon.png',             // 图标文件路径
        }),
      }),

      // 起点标记样式（使用 SVG 图标）
      'startMarker': new Style({
        image: new Icon({
          anchor: [0.5, 0.5],               // 锚点位置（中心点）
          src: 'start-marker.svg',          // SVG 图标文件
          scale: 1.2,                       // 缩放比例
        }),
      }),

      // 动画标记样式
      'geoMarker': new Style({
        image: new CircleStyle({
          radius: 7,                        // 圆形半径
          fill: new Fill({color: 'black'}), // 填充颜色
          stroke: new Stroke({
            color: 'white',                // 描边颜色
            width: 2,                       // 描边宽度
          }),
        }),
      }),
    };

    // =============================================================================
    // 7. 创建矢量图层并添加要素
    // =============================================================================

    const vectorLayer = new VectorLayer({
      // 创建矢量数据源
      source: new VectorSource({
        // 添加所有要素到数据源
        features: [routeFeature, geoMarker, startMarker, endMarker, fixedStartMarker],
      }),

      // 动态样式函数：根据要素类型返回对应的样式
      style: function (feature) {
        return styles[feature.get('type')];
      },
    });

    // 将矢量图层添加到地图
    map.addLayer(vectorLayer);

    // =============================================================================
    // 8. 动画控制逻辑
    // =============================================================================

    // 获取控制元素
    const speedInput = document.getElementById('speed');        // 速度滑块
    const speedValue = document.getElementById('speed-value');  // 速度显示文本
    const startButton = document.getElementById('start-animation'); // 开始/停止按钮

    // 动画状态变量
    let animating = false;      // 动画是否正在运行
    let distance = 0;           // 动画进度（0-2，0和2表示起点，1表示终点）
    let lastTime;              // 上一帧的时间戳

    // 监听速度滑块变化，实时更新显示值
    speedInput.addEventListener('input', function() {
      speedValue.textContent = speedInput.value;
    });

    // =============================================================================
    // 9. 动画核心函数
    // =============================================================================

    /**
     * 移动动画标记的函数
     * 这个函数会在每一帧渲染时被调用
     * @param {Object} event - 渲染事件对象
     */
    function moveFeature(event) {
      // 获取当前速度设置
      const speed = Number(speedInput.value);

      // 获取当前时间戳
      const time = event.frameState.time;

      // 计算距离上一帧的时间差（毫秒）
      const elapsedTime = time - lastTime;

      // 更新动画进度
      // 公式说明：speed * elapsedTime / 1e6 控制动画速度
      // % 2 确保进度在 0-2 之间循环
      distance = (distance + (speed * elapsedTime) / 1e6) % 2;
      lastTime = time;

      // 根据进度计算当前位置
      // distance > 1 ? 2 - distance : distance 实现往返动画
      // 当 distance > 1 时，从终点返回起点
      const currentCoordinate = route.getCoordinateAt(
        distance > 1 ? 2 - distance : distance,
      );

      // 更新动画标记位置
      position.setCoordinates(currentCoordinate);

      // 获取矢量渲染上下文
      const vectorContext = getVectorContext(event);

      // 设置动画标记样式
      vectorContext.setStyle(styles.geoMarker);

      // 绘制动画标记
      vectorContext.drawGeometry(position);

      // 请求 OpenLayers 继续下一帧渲染
      // 这是实现连续动画的关键
      map.render();
    }

    // =============================================================================
    // 10. 动画控制函数
    // =============================================================================

    /**
     * 开始动画
     */
    function startAnimation() {
      animating = true;
      lastTime = Date.now();                    // 记录开始时间
      startButton.textContent = '停止动画';      // 更新按钮文本

      // 注册 postrender 事件监听器
      // 这个事件会在每一帧渲染完成后触发
      vectorLayer.on('postrender', moveFeature);

      // 隐藏静态的动画标记，触发重绘
      geoMarker.setGeometry(null);
    }

    /**
     * 停止动画
     */
    function stopAnimation() {
      animating = false;
      startButton.textContent = '开始动画';      // 恢复按钮文本

      // 将动画标记固定在当前位置
      geoMarker.setGeometry(position);

      // 移除 postrender 事件监听器，停止动画
      vectorLayer.un('postrender', moveFeature);
    }

    // =============================================================================
    // 11. 用户交互事件
    // =============================================================================

    // 绑定按钮点击事件
    startButton.addEventListener('click', function () {
      if (animating) {
        stopAnimation();  // 如果正在运行，则停止
      } else {
        startAnimation(); // 如果已停止，则开始
      }
    });

  });
});

/**
 * 教学要点总结：
 *
 * 1. 地图初始化：使用 Map、View、Layer 构建 OpenLayers 应用基础
 * 2. 图层管理：叠加瓦片图层和矢量图层实现丰富的地图效果
 * 3. 数据加载：使用 GeoJSON 格式加载外部地理数据
 * 4. 样式定制：为不同类型的要素定义不同的视觉样式
 * 5. 动画实现：利用 postrender 事件实现平滑的轨迹动画
 * 6. 用户交互：通过 DOM 事件实现用户控制功能
 * 7. 坐标系统：理解 Web Mercator 投影和经纬度坐标的转换
 * 8. 性能优化：使用事件监听和渲染循环实现高效动画
 *
 * 扩展建议：
 * - 添加更多轨迹数据的格式支持（KML、GPX 等）
 * - 实现轨迹编辑功能
 * - 添加多个轨迹同时显示
 * - 集成更多地图服务（百度地图、高德地图等）
 * - 添加轨迹统计信息（距离、时间、速度等）
 */