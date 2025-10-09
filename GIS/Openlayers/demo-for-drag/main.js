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

const map = new Map({
  layers: [
    tiandituLayer,
    tiandituAnnotationLayer,
    markerLayer
  ],
  target: 'map',
  view: new View({
    center: fromLonLat(xiAnCoords), // 中心设置为西安
    zoom: 6,
    minZoom: 1,
    maxZoom: 18
  }),
});

// 地图切换功能
const tiandituBtn = document.getElementById('map-tianditu');
const osmBtn = document.getElementById('map-osm');

// 切换到天地图
function switchToTianditu() {
  map.getLayers().remove(osmLayer);
  map.getLayers().insertAt(0, tiandituLayer);
  map.getLayers().insertAt(1, tiandituAnnotationLayer);

  tiandituBtn.classList.add('active');
  osmBtn.classList.remove('active');
}

// 切换到OSM地图
function switchToOSM() {
  map.getLayers().remove(tiandituLayer);
  map.getLayers().remove(tiandituAnnotationLayer);
  map.getLayers().insertAt(0, osmLayer);

  osmBtn.classList.add('active');
  tiandituBtn.classList.remove('active');
}

// 绑定按钮事件
tiandituBtn.addEventListener('click', switchToTianditu);
osmBtn.addEventListener('click', switchToOSM);

const extractStyles = document.getElementById('extractstyles');
let dragAndDropInteraction;

function setInteraction() {
  if (dragAndDropInteraction) {
    map.removeInteraction(dragAndDropInteraction);
  }
  dragAndDropInteraction = new DragAndDrop({
    formatConstructors: [
      GPX,
      GeoJSON,
      IGC,
      // use constructed format to set options
      new KML({extractStyles: extractStyles.checked}),
      TopoJSON,
    ],
  });
  dragAndDropInteraction.on('addfeatures', function (event) {
    const vectorSource = new VectorSource({
      features: event.features,
    });
    map.addLayer(
      new VectorLayer({
        source: vectorSource,
      }),
    );
    map.getView().fit(vectorSource.getExtent());
  });
  map.addInteraction(dragAndDropInteraction);
}
setInteraction();

extractStyles.addEventListener('change', setInteraction);

const displayFeatureInfo = function (pixel) {
  const features = [];
  map.forEachFeatureAtPixel(pixel, function (feature) {
    features.push(feature);
  });
  if (features.length > 0) {
    const info = [];
    let i, ii;
    for (i = 0, ii = features.length; i < ii; ++i) {
      info.push(features[i].get('name'));
    }
    document.getElementById('info').innerHTML = info.join(', ') || '&nbsp';
  } else {
    document.getElementById('info').innerHTML = '&nbsp;';
  }
};

map.on('pointermove', function (evt) {
  if (evt.dragging) {
    return;
  }
  displayFeatureInfo(evt.pixel);
});

map.on('click', function (evt) {
  displayFeatureInfo(evt.pixel);
});

// Sample data downloads

const link = document.getElementById('download');

function download(fullpath, filename) {
  fetch(fullpath)
    .then(function (response) {
      return response.blob();
    })
    .then(function (blob) {
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
    });
}

document.getElementById('download-gpx').addEventListener('click', function () {
  download('data/gpx/fells_loop.gpx', 'fells_loop.gpx');
});

document
  .getElementById('download-geojson')
  .addEventListener('click', function () {
    download('data/geojson/roads-seoul.geojson', 'roads-seoul.geojson');
  });

document.getElementById('download-igc').addEventListener('click', function () {
  download('data/igc/Ulrich-Prinz.igc', 'Ulrich-Prinz.igc');
});

document.getElementById('download-kml').addEventListener('click', function () {
  download('data/kml/states.kml', 'states.kml');
});

document
  .getElementById('download-topojson')
  .addEventListener('click', function () {
    download('data/topojson/fr-departments.json', 'fr-departments.json');
  });

