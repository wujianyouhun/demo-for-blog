# GeoAI 数据下载工具

## 项目介绍

GeoAI 数据下载工具是一个用于从多个数据源获取地理空间数据的工具，支持以下功能：

- **NAIP 影像下载**：从微软 Planetary Computer 下载美国国家农业影像计划 (NAIP) 数据
- **Overture Maps 数据下载**：从 Overture Maps 下载建筑物、地址等开放地理空间数据
- **Sentinel-2 数据下载**：从 Planetary Computer 下载 Sentinel-2 卫星影像
- **Landsat-8 数据下载**：从 Planetary Computer 下载 Landsat-8 卫星影像

## 环境配置

### 系统要求

- Python 3.7 或更高版本
- 稳定的网络连接

### 安装依赖

1. 进入项目目录：
   ```bash
   ..\geoai\data-downlad
   ```
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 基本使用方法

### 运行程序

```bash
python download_data.py
```

运行后会显示交互式菜单，您可以选择要下载的数据类型：

```
============================================================
GeoAI 数据下载工具
============================================================

请选择要下载的数据类型:
1. NAIP 影像 (美国国家农业影像计划)
2. Overture 建筑物数据
3. Sentinel-2 数据 (STAC)
4. Landsat-8 数据 (STAC)
5. 退出
------------------------------------------------------------
请输入选项 (1-5): 
```

## 下载指定地点和时间的数据

### 方法一：通过交互式菜单下载

选择相应的数据类型后，程序会使用默认的边界框和时间范围进行下载。

### 方法二：修改代码下载指定地点和时间

如果您需要下载特定地点和时间的数据，可以修改 `download_data.py` 文件中的参数：

#### 1. 下载 NAIP 影像（指定地点和时间）

**步骤**：

1. 打开 `download_data.py` 文件
2. 找到 `main()` 函数中的 NAIP 下载部分（约 420-437 行）
3. 修改以下参数：
   - `bbox`：设置为您需要的边界框 \[min\_lon, min\_lat, max\_lon, max\_lat]
   - `year`：设置为您需要的年份

**示例**：

```python
# 修改前
bbox = (-122.51, 37.71, -122.41, 37.81)  # 旧金山区域
files = download_naip(
    bbox=bbox,
    output_dir="naip_data",
    year=2020,  # 2020年数据
    max_items=5,
    overwrite=False,
    preview=True
)

# 修改后（下载纽约区域 2021 年数据）
bbox = (-74.05, 40.65, -73.95, 40.75)  # 纽约区域
files = download_naip(
    bbox=bbox,
    output_dir="naip_data_newyork",
    year=2021,  # 2021年数据
    max_items=5,
    overwrite=False,
    preview=True
)
```

#### 2. 下载 Overture 建筑物数据（指定地点）

**步骤**：

1. 打开 `download_data.py` 文件
2. 找到 `main()` 函数中的 Overture 下载部分（约 439-453 行）
3. 修改 `bbox` 参数为您需要的边界框

**示例**：

```python
# 修改前
bbox = (-74.01, 40.70, -73.99, 40.72)  # 曼哈顿区域

# 修改后（下载芝加哥区域）
bbox = (-87.64, 41.87, -87.62, 41.89)  # 芝加哥区域
```

#### 3. 下载 Sentinel-2 数据（指定地点和时间）

**步骤**：

1. 打开 `download_data.py` 文件
2. 找到 `main()` 函数中的 Sentinel-2 下载部分（约 455-484 行）
3. 修改以下参数：
   - `bbox`：设置为您需要的边界框
   - `time_range`：设置为您需要的时间范围，格式为 "start/end"

**示例**：

```python
# 修改前
bbox = [-122.4, 37.7, -122.3, 37.8]  # 旧金山区域
items = pc_stac_search(
    collection="sentinel-2-l2a",
    bbox=bbox,
    time_range="2023-01-01/2023-01-31",  # 2023年1月
    max_items=2
)

# 修改后（下载洛杉矶 2023 年 6 月数据）
bbox = [-118.3, 34.0, -118.2, 34.1]  # 洛杉矶区域
items = pc_stac_search(
    collection="sentinel-2-l2a",
    bbox=bbox,
    time_range="2023-06-01/2023-06-30",  # 2023年6月
    max_items=2
)
```

#### 4. 下载 Landsat-8 数据（指定地点和时间）

**步骤**：

1. 打开 `download_data.py` 文件
2. 找到 `main()` 函数中的 Landsat-8 下载部分（约 486-515 行）
3. 修改以下参数：
   - `bbox`：设置为您需要的边界框
   - `time_range`：设置为您需要的时间范围

**示例**：

```python
# 修改前
bbox = [-122.4, 37.7, -122.3, 37.8]  # 旧金山区域
items = pc_stac_search(
    collection="landsat-c2-l2",
    bbox=bbox,
    time_range="2023-01-01/2023-01-31",  # 2023年1月
    max_items=1
)

# 修改后（下载迈阿密 2023 年 3 月数据）
bbox = [-80.2, 25.7, -80.1, 25.8]  # 迈阿密区域
items = pc_stac_search(
    collection="landsat-c2-l2",
    bbox=bbox,
    time_range="2023-03-01/2023-03-31",  # 2023年3月
    max_items=1
)
```

## 数据保存位置

所有下载的数据都保存在项目根目录的 `data` 文件夹中：

- NAIP 影像：`data/naip_data/`
- Overture 建筑物数据：`data/overture_buildings/`
- Sentinel-2 数据：`data/sentinel2_data/`
- Landsat-8 数据：`data/landsat8_data/`

## 常见问题和注意事项

### 1. 下载速度慢

**解决方案**：

- 增加 `max_workers` 参数值（默认为 1）
- 使用较小的 `max_items` 值，分批下载
- 确保网络连接稳定

### 2. 内存不足

**解决方案**：

- 减少 `max_items` 值
- 避免一次性下载太多资产
- 使用较小的边界框

### 3. 认证错误

**解决方案**：

- 确保 Planetary Computer 认证正确
- 检查网络连接和 API 访问权限

### 4. Overture Maps 依赖错误

**解决方案**：

- 安装 overturemaps 包：
  ```bash
  pip install overturemaps
  ```

### 5. 数据格式问题

**解决方案**：

- 确保输出目录存在且有写入权限
- 检查文件扩展名是否正确

## 示例：下载北京地区的数据

### 下载 Sentinel-2 数据（北京地区）

1. 修改 `download_data.py` 文件中的 Sentinel-2 下载部分：

```python
# 北京区域边界框
bbox = [116.3, 39.9, 116.4, 40.0]
items = pc_stac_search(
    collection="sentinel-2-l2a",
    bbox=bbox,
    time_range="2023-05-01/2023-05-31",  # 2023年5月
    max_items=2
)
```

1. 运行程序并选择选项 3
2. 数据将保存在 `data/sentinel2_data/` 目录中

## 技术支持

如果您在使用过程中遇到问题，请检查以下几点：

1. 确保所有依赖包已正确安装
2. 检查网络连接是否稳定
3. 确保边界框格式正确 \[min\_lon, min\_lat, max\_lon, max\_lat]
4. 时间范围格式正确 "start/end"（如 "2023-01-01/2023-01-31"）

##
