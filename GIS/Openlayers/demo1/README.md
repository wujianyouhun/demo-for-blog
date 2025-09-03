# OpenLayers 地图应用教学Demo

> 一个完整的OpenLayers地图应用示例，适合初学者学习WebGIS开发

## 📖 项目简介

本项目是一个基于OpenLayers 的完整地图应用Demo，展示了如何创建具有交互功能的地图应用。项目包含地图显示、地理数据加载、用户交互等功能，是学习OpenLayers的理想起点。

### ✨ 主要功能

- 🗺️ 显示OpenStreetMap底图
- 📍 加载并显示临泽县GeoJSON地理数据
- 🔍 地图缩放和定位功能
- 🎯 快速定位到临泽县和西安市
- 📱 响应式设计，支持全屏显示

### 🎯 学习目标

- 理解OpenLayers的基本架构
- 掌握地图、视图、图层、数据源的概念
- 学会加载和显示地理数据
- 实现地图交互功能（缩放、居中、定位）
- 理解坐标系统和投影

## 👨‍💻 作者信息

**学GIS的小宝同学**

- 🎓 GIS专业学习者
- 🌍 热爱地理信息科学
- 💻 专注于GIS+开发
- 📚 致力于分享GIS学习经验

## 🚀 快速开始

### 环境要求

- Node.js >= 14.0.0
- npm >= 6.0.0 或 yarn >= 1.22.0
- 现代浏览器（Chrome、Firefox、Safari、Edge）

### 安装依赖

```bash
# 使用npm
npm install

# 或使用yarn
yarn install
```

### 开发模式运行

```bash
# 启动开发服务器
npm run dev

# 或使用yarn
yarn dev
```

开发服务器启动后，在浏览器中访问 `http://localhost:3000` 即可查看应用。

### 构建生产版本

```bash
# 构建生产版本
npm run build

# 或使用yarn
yarn build
```

构建完成后，生产文件将生成在 `dist` 目录中。

### 预览生产版本

```bash
# 预览构建后的文件
npm run preview

# 或使用yarn
yarn preview
```

## 📁 项目结构

```
demo1/
├── index.html              # 主页面文件
├── main.js                 # 主要JavaScript代码
├── linze.geojson          # 临泽县地理数据
├── package.json            # 项目配置文件
├── package-lock.json       # 依赖锁定文件
├── node_modules/           # 依赖包目录
└── README.md              # 项目说明文档
```

## 🛠️ 技术栈

- **地图引擎**: OpenLayers 6
- **构建工具**: Vite
- **包管理**: npm/yarn
- **数据格式**: GeoJSON
- **地图服务**: OpenStreetMap

## 📚 核心概念

### 1. 数据源（Source）
- 管理地理数据的加载和存储
- 提供数据查询和过滤功能
- 处理数据的更新和同步

### 2. 图层（Layer）
- 控制数据的显示方式
- 设置数据的样式（颜色、大小、透明度等）
- 管理数据的可见性和交互性

### 3. 视图（View）
- 控制地图显示的中心点
- 设置地图的缩放级别
- 定义坐标系统（投影）

### 4. 地图（Map）
- 整合所有图层和视图
- 管理用户交互（鼠标、键盘、触摸）
- 提供地图控制工具

## 🎮 功能说明

### 缩放到临泽区域
- 自动调整地图视图以显示临泽县的完整地理边界
- 使用`view.fit()`方法自动计算最佳缩放级别和中心点

### 缩放到临泽县
- 将地图中心移动到临泽县的坐标位置
- 设置合适的缩放级别（10级，适合查看县级区域）
- 使用动画效果，提供流畅的用户体验

### 居中到西安市
- 将地图中心移动到西安市位置
- 保持当前的缩放级别不变
- 只改变位置，不改变缩放

## 🔧 自定义配置

### 修改城市坐标

在 `main.js` 文件中修改 `CITY_COORDINATES` 对象：

```javascript
const CITY_COORDINATES = {
  linze: [100.497777, 39.480275],      // 临泽县坐标
  xian: [108.948024, 34.263161]        // 西安市坐标
};
```

### 修改地图样式

在 `main.js` 文件中修改 `vectorLayer` 的样式配置：

```javascript
style: {
  'fill-color': 'rgba(255, 255, 255, 0.6)',      // 填充颜色
  'stroke-width': 1,                              // 边框宽度
  'stroke-color': '#319FD3',                       // 边框颜色
  // ... 更多样式配置
}
```

### 修改初始视图

在 `main.js` 文件中修改 `view` 的配置：

```javascript
const view = new View({
  center: [100, 40],                        // 初始中心点
  zoom: 4,                                  // 初始缩放级别
  projection: 'EPSG:4326'                   // 坐标系统
});
```

## 📦 部署说明

### 静态文件部署

构建完成后，将 `dist` 目录中的文件上传到任何静态文件服务器即可。

### 支持的服务商

- GitHub Pages
- Netlify
- Vercel
- 阿里云OSS
- 腾讯云COS
- 七牛云

### 部署到GitHub Pages

1. 在GitHub上创建仓库
2. 推送代码到仓库
3. 在仓库设置中启用GitHub Pages
4. 选择 `dist` 目录作为源目录

## 🐛 常见问题

### Q: 地图无法显示？
A: 检查网络连接，确保能够访问OpenStreetMap服务。

### Q: GeoJSON数据加载失败？
A: 确保 `linze.geojson` 文件存在且格式正确。

### Q: 按钮点击无反应？
A: 打开浏览器开发者工具，查看控制台是否有错误信息。

### Q: 坐标定位不准确？
A: 确认坐标格式为 `[经度, 纬度]`，且使用WGS84坐标系统。

## 📖 学习资源

- [OpenLayers官方文档](https://openlayers.org/)
- [OpenLayers API参考](https://openlayers.org/en/latest/apidoc/)
- [GeoJSON规范](https://geojson.org/)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 贡献步骤




## 📞 联系方式

- 📧 邮箱: xiaobaogis@163.com
- 🐦 博客: [@学GIS的小宝同学](https://weibo.com/example)
- 💬 QQ群: 505180987


