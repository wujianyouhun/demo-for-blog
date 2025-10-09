// =============================================================================
// OpenLayers地图应用 - 拖拽上传示例
// 作者：小宝同学（GIS学习者）
// 功能：展示如何使用OpenLayers实现地图展示、拖拽上传数据、地图切换等功能
// =============================================================================

// 导入OpenLayers核心组件
import Map from 'ol/Map.js';              // 地图主类
import View from 'ol/View.js';            // 地图视图类
import GPX from 'ol/format/GPX.js';       // GPX格式解析器
import GeoJSON from 'ol/format/GeoJSON.js'; // GeoJSON格式解析器
import IGC from 'ol/format/IGC.js';       // IGC格式解析器
import KML from 'ol/format/KML.js';       // KML格式解析器
import TopoJSON from 'ol/format/TopoJSON.js'; // TopoJSON格式解析器
import DragAndDrop from 'ol/interaction/DragAndDrop.js'; // 拖拽交互
import TileLayer from 'ol/layer/Tile.js';  // 瓦片图层类
import VectorLayer from 'ol/layer/Vector.js'; // 矢量图层类
import VectorSource from 'ol/source/Vector.js'; // 矢量数据源
import TileImage from 'ol/source/TileImage.js'; // 瓦片图片数据源
import OSM from 'ol/source/OSM.js';        // OpenStreetMap数据源
import Feature from 'ol/Feature.js';       // 地理要素类
import Point from 'ol/geom/Point.js';     // 点几何类
import {fromLonLat, toLonLat} from 'ol/proj.js'; // 坐标转换工具
import {getWidth} from 'ol/extent.js';     // 范围计算工具
import {Style, Circle, Fill, Stroke} from 'ol/style.js'; // 样式组件

// =============================================================================
// 1. 地图数据源配置
// =============================================================================

// 天地图API密钥（从天地图官网申请）
// 注意：实际项目中应该将密钥保存在环境变量或配置文件中
const tk = 'f4d0553a23372a2f48c74851c7e46f4d';

// 创建天地图影像图层
// 影像图层显示卫星图像或地形图
const tiandituLayer = new TileLayer({
  source: new TileImage({
    projection: 'EPSG:3857',  // 使用Web墨卡托投影
    url: 'http://t{0-7}.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=' + tk,
    // URL说明：
    // - t{0-7}: 使用8个子域名进行负载均衡
    // - T=img_w: 影像图层类型（w表示Web墨卡托投影）
    // - x,y,z: 瓦片坐标
    // - tk: API密钥
  })
});

// 创建天地图注记图层
// 注记图层显示地名、道路名称等文字信息
const tiandituAnnotationLayer = new TileLayer({
  source: new TileImage({
    projection: 'EPSG:3857',
    url: 'http://t{0-7}.tianditu.gov.cn/DataServer?T=cia_w&x={x}&y={y}&l={z}&tk=' + tk,
    // T=cia_w: 影像注记图层类型
  })
});

// 创建OpenStreetMap图层
// OSM是一个开源的免费世界地图
const osmLayer = new TileLayer({
  source: new OSM()  // OSM数据源，无需额外配置
});

// =============================================================================
// 2. 地图要素和样式配置
// =============================================================================

// 创建西安城市标记点
// 坐标说明：[经度, 纬度] 格式，这是WGS84坐标系（EPSG:4326）
const xiAnCoords = [108.940174, 34.341568]; // 西安市中心坐标

// 创建一个地理要素（Feature）
// Feature是OpenLayers中的基本地理对象，可以包含几何图形和属性
const xiAnFeature = new Feature({
  geometry: new Point(fromLonLat(xiAnCoords)),  // 将经纬度转换为Web墨卡托坐标
  name: '西安',                                  // 要素属性：城市名称
  population: '1000万+',                         // 要素属性：人口
  type: '省会城市'                              // 要素属性：城市类型
});

// 创建标记点样式
// Style定义了要素的显示外观
const xiAnStyle = new Style({
  image: new Circle({        // 使用圆形作为标记样式
    radius: 8,               // 圆形半径（像素）
    fill: new Fill({         // 填充样式
      color: '#ff0000'       // 红色填充
    }),
    stroke: new Stroke({     // 边框样式
      color: '#ffffff',      // 白色边框
      width: 2               // 边框宽度
    })
  })
});

// 将样式应用到西安标记要素
xiAnFeature.setStyle(xiAnStyle);

// 创建矢量图层来显示标记要素
// VectorLayer用于显示矢量数据（点、线、面）
const markerLayer = new VectorLayer({
  source: new VectorSource({  // 矢量数据源
    features: [xiAnFeature]  // 包含的要素数组
  }),
  // 图层属性
  properties: {
    name: '城市标记图层',
    type: 'marker'
  }
});

// =============================================================================
// 3. 创建地图实例
// =============================================================================

// 创建地图主实例
// Map是OpenLayers的核心类，负责管理图层、交互和视图
const map = new Map({
  layers: [  // 图层配置（从下往上叠加显示）
    tiandituLayer,           // 底层：天地图影像
    tiandituAnnotationLayer, // 中层：天地图注记
    markerLayer             // 顶层：标记点图层
  ],
  target: 'map',  // 地图容器的DOM元素ID
  view: new View({  // 地图视图配置
    center: fromLonLat(xiAnCoords), // 地图中心点（转换为Web墨卡托坐标）
    zoom: 6,                          // 初始缩放级别
    minZoom: 1,                      // 最小缩放级别
    maxZoom: 18,                     // 最大缩放级别
    projection: 'EPSG:3857'          // 使用Web墨卡托投影
  }),
  // 地图控件（默认包含缩放控件）
  controls: []  // 清空默认控件，使用自定义控制面板
});

// =============================================================================
// 4. 地图切换功能
// =============================================================================

// 获取地图切换按钮DOM元素
const tiandituBtn = document.getElementById('map-tianditu');
const osmBtn = document.getElementById('map-osm');

// 切换到天地图函数
function switchToTianditu() {
  // 移除OSM图层（如果存在）
  map.getLayers().remove(osmLayer);

  // 添加天地图图层（确保顺序正确）
  map.getLayers().insertAt(0, tiandituLayer);           // 底层：影像
  map.getLayers().insertAt(1, tiandituAnnotationLayer); // 顶层：注记

  // 更新按钮状态
  tiandituBtn.classList.add('active');
  osmBtn.classList.remove('active');
}

// 切换到OSM地图函数
function switchToOSM() {
  // 移除天地图图层
  map.getLayers().remove(tiandituLayer);
  map.getLayers().remove(tiandituAnnotationLayer);

  // 添加OSM图层
  map.getLayers().insertAt(0, osmLayer);

  // 更新按钮状态
  osmBtn.classList.add('active');
  tiandituBtn.classList.remove('active');
}

// 为地图切换按钮绑定点击事件
tiandituBtn.addEventListener('click', switchToTianditu);
osmBtn.addEventListener('click', switchToOSM);

// =============================================================================
// 5. 拖拽上传功能
// =============================================================================

// 获取"提取样式"复选框元素
const extractStyles = document.getElementById('extractstyles');
let dragAndDropInteraction;  // 拖拽交互对象

// 设置拖拽交互功能
function setInteraction() {
  // 如果已存在拖拽交互，先移除
  if (dragAndDropInteraction) {
    map.removeInteraction(dragAndDropInteraction);
  }

  // 创建新的拖拽交互
  dragAndDropInteraction = new DragAndDrop({
    formatConstructors: [  // 支持的文件格式
      GPX,                                  // GPX轨迹格式
      GeoJSON,                              // GeoJSON地理数据格式
      IGC,                                  // IGC滑翔机轨迹格式
      // KML格式需要构造函数来设置选项
      new KML({extractStyles: extractStyles.checked}),  // 是否提取KML中的样式
      TopoJSON,                             // TopoJSON拓扑格式
    ],
  });

  // 监听要素添加事件（当拖拽文件成功时触发）
  dragAndDropInteraction.on('addfeatures', function (event) {
    // 创建新的矢量数据源来存储拖拽的要素
    const vectorSource = new VectorSource({
      features: event.features,  // 从文件中解析出的要素
    });

    // 创建新的矢量图层并添加到地图
    map.addLayer(
      new VectorLayer({
        source: vectorSource,
        style: function(feature) {
          // 可以为不同类型的要素设置不同的样式
          return feature.getStyle() || new Style({
            stroke: new Stroke({
              color: '#3388ff',
              width: 2
            }),
            fill: new Fill({
              color: 'rgba(51, 136, 255, 0.2)'
            })
          });
        }
      }),
    );

    // 调整地图视图以显示所有新添加的要素
    map.getView().fit(vectorSource.getExtent(), {
      padding: [50, 50, 50, 50],  // 设置边距，避免要素紧贴边缘
      duration: 1000              // 动画持续时间（毫秒）
    });
  });

  // 将拖拽交互添加到地图
  map.addInteraction(dragAndDropInteraction);
}

// 初始化拖拽交互
setInteraction();

// 监听"提取样式"复选框变化，重新设置拖拽交互
extractStyles.addEventListener('change', setInteraction);

// =============================================================================
// 6. 地图交互功能
// =============================================================================

// 显示要素信息函数
// 当鼠标悬停或点击要素时，显示要素的属性信息
const displayFeatureInfo = function (pixel) {
  const features = [];  // 存储鼠标位置的要素

  // 获取指定像素位置的所有要素
  map.forEachFeatureAtPixel(pixel, function (feature) {
    features.push(feature);
  });

  if (features.length > 0) {
    // 如果找到要素，显示它们的属性信息
    const info = [];
    let i, ii;
    for (i = 0, ii = features.length; i < ii; ++i) {
      const feature = features[i];
      const name = feature.get('name');
      const type = feature.get('type');
      const population = feature.get('population');

      // 构建要素信息字符串
      let featureInfo = name || '未命名要素';
      if (type) featureInfo += ` (${type})`;
      if (population) featureInfo += ` - ${population}`;

      info.push(featureInfo);
    }
    document.getElementById('info').innerHTML = info.join('<br>') || '&nbsp';
  } else {
    // 如果没有找到要素，显示空内容
    document.getElementById('info').innerHTML = '暂无要素信息';
  }
};

// 监听鼠标移动事件
map.on('pointermove', function (evt) {
  if (evt.dragging) {
    return;  // 如果正在拖拽，不处理
  }
  displayFeatureInfo(evt.pixel);  // 显示鼠标位置的要素信息
});

// 监听鼠标点击事件
map.on('click', function (evt) {
  displayFeatureInfo(evt.pixel);  // 显示点击位置的要素信息
});

// =============================================================================
// 7. 示例数据下载功能
// =============================================================================

// 获取下载链接元素
const link = document.getElementById('download');

// 文件下载函数
// 使用Fetch API获取文件并触发下载
function download(fullpath, filename) {
  fetch(fullpath)
    .then(function (response) {
      return response.blob();  // 将响应转换为Blob对象
    })
    .then(function (blob) {
      // 创建下载链接
      link.href = URL.createObjectURL(blob);  // 创建Blob URL
      link.download = filename;               // 设置下载文件名
      link.click();                           // 触发点击下载
    })
    .catch(function (error) {
      console.error('文件下载失败:', error);
      alert('文件下载失败，请检查网络连接');
    });
}

// 为各种格式的下载按钮绑定事件
document.getElementById('download-gpx').addEventListener('click', function () {
  download('data/gpx/fells_loop.gpx', 'fells_loop.gpx');
  // GPX: GPS Exchange Format，常用于存储GPS轨迹和航点
});

document.getElementById('download-geojson').addEventListener('click', function () {
  download('data/geojson/roads-seoul.geojson', 'roads-seoul.geojson');
  // GeoJSON: 地理数据交换格式，基于JSON
});

document.getElementById('download-igc').addEventListener('click', function () {
  download('data/igc/Ulrich-Prinz.igc', 'Ulrich-Prinz.igc');
  // IGC: 滑翔机飞行记录格式
});

document.getElementById('download-kml').addEventListener('click', function () {
  download('data/kml/states.kml', 'states.kml');
  // KML: Keyhole Markup Language，Google Earth使用的格式
});

document.getElementById('download-topojson').addEventListener('click', function () {
  download('data/topojson/fr-departments.json', 'fr-departments.json');
  // TopoJSON: 拓扑JSON格式，可以更高效地存储地理数据
});

