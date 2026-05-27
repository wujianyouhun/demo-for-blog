# GLTF 3D 模型查看器

一个功能完整的 GLTF/GLB 3D 模型查看器，支持模型量测和属性查看。

## 📦 功能特性

- 🎨 **模型查看** - 加载和显示 GLTF/GLB 3D 模型
- 📏 **距离测量** - 在模型表面点击两个点测量距离
- 📐 **面积测量** - 点击三个点测量三角形面积
- 📊 **属性查看** - 查看模型的几何尺寸、顶点数、三角数等信息
- 🎮 **交互式控制** - 旋转、平移、缩放

## 🚀 快速开始

### ✅ 直接双击打开（推荐）

**现在可以直接双击 `viewer.html` 文件打开使用！**

所有 Three.js 库已下载到本地，无需网络连接即可使用。

### 项目结构
```
gltf-viewer/
├── viewer.html              # 主页面（直接双击打开）
├── scene_3d.glb             # 示例模型（自动加载）
├── libs/                    # Three.js 库文件（本地）
│   ├── three.min.js         # Three.js 核心库
│   ├── controls/
│   │   └── OrbitControls.js
│   └── loaders/
│       └── GLTFLoader.js
└── README.md               # 说明文档
```

## 📖 使用说明

### 加载模型

- 页面打开时**自动加载**目录下的 `scene_3d.glb`
- 点击 **"📂 打开模型文件"** 选择本地 GLTF/GLB 文件
- 点击 **"🔄 加载示例模型"** 重新加载 `scene_3d.glb`

### 量测工具

1. 点击 **"测量距离"** 或 **"测量面积"**
2. 在模型表面点击选择点
3. 结果会显示在面板中
4. 点击 **"清除"** 重置量测

### 视图控制

- 左键拖拽：旋转视角
- 右键拖拽：平移
- 滚轮：缩放

## 📚 技术栈

- Three.js r139
- 纯 JavaScript（全局 THREE 对象）
- 原生 HTML/CSS

## 🔧 库文件说明

所有 Three.js 库文件已下载到 `libs/` 目录：

- `libs/three.min.js` - Three.js 核心库
- `libs/controls/OrbitControls.js` - 轨道控制器
- `libs/loaders/GLTFLoader.js` - GLTF 加载器

## ✅ 无需网络

项目完全本地化，所有依赖都已下载，**无需网络连接**即可使用！
