# 数据下载文档

## 数据来源

本项目使用 **Sentinel-2** 卫星影像数据, 通过 **Microsoft Planetary Computer** 获取。

### Sentinel-2 特点

- 空间分辨率: 10m / 20m / 60m (多波段)
- 时间分辨率: 5 天重访
- 光谱波段: 13 个波段
- 覆盖范围: 全球

## 支持区域

| 区域 | 标识 | 范围 (lon/lat) |
|------|------|----------------|
| 北京 | beijing | [116.2, 39.7, 116.6, 40.1] |
| 上海 | shanghai | [121.3, 31.0, 121.7, 31.4] |
| 深圳 | shenzhen | [113.8, 22.4, 114.2, 22.7] |
| 成都 | chengdu | [103.9, 30.5, 104.3, 30.8] |
| 武汉 | wuhan | [114.1, 30.4, 114.5, 30.7] |

## 下载方式

### CLI 方式

```bash
# 下载北京地区 2023 年 6 月数据
python scripts/prepare_data.py --region beijing --date 2023-06-01

# 下载上海地区数据, 指定输出目录
python scripts/prepare_data.py --region shanghai --date 2023-07-01 --output data/shanghai
```

### Web 界面方式

1. 打开前端页面, 切换到 "数据管理" 标签
2. 选择预设区域或输入自定义范围
3. 设置日期范围和云量上限
4. 点击 "下载数据"

## 数据格式

下载的数据保存在 `data/<region>/` 目录下:

```
data/
└── beijing/
    ├── images/
    │   └── sentinel2_北京_2023-06-01.npy  # 13波段影像
    ├── labels/
    │   └── labels.npy                       # 6类别标签
    └── metadata.json                        # 元数据
```

## 类别定义

| ID | 英文名 | 中文名 | 颜色 |
|----|--------|--------|------|
| 0 | background | 背景 | #808080 |
| 1 | building | 建筑 | #e6194b |
| 2 | road | 道路 | #ffe119 |
| 3 | water | 水体 | #3cb44b |
| 4 | vegetation | 植被 | #42d4f4 |
| 5 | barren | 裸地 | #f58231 |

## API 接口

```
POST /api/data/download
{
  "region": "beijing",
  "bbox": null,
  "date_start": "2023-06-01",
  "date_end": "2023-08-31",
  "cloud_cover_max": 20
}

GET /api/data/download/status/{task_id}
GET /api/data/files
```
