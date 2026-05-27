# 使用pygeoai库制作标签数据

<br />

<br />

## 资料准备

### 数据准备

现有STAC标准数据管理平台，访问地址 <http://localhost:5173/>

高分辨率影像数据，STAC  item\_url:http\://localhost:8010/collections/raster-default/items/0dcc269fd72a4c4e8a3c388564122fd1；为本次样本制作主要影像数据。覆盖范围较小，为西安市部分地区。

建筑shp数据： STAC item\_url:http\://localhost:8010/collections/vector-default/items/90dbe2fc21b84d83891b87fd61ed24e6覆盖范围为全国数据。平台管理shp数据目前是一zip方式管理。

影像和SHP的坐标系都是地理坐标系，“EPSG"4326”

### 系统环境

conda geoai 环境好提供相应的geoai-py库。

<br />

## 数据处理要求

1. 要求将目标影像和建筑shp对齐，包含参照系和空间范围，生产结果存放在data 文件夹中，已准备下一步标签的裁切和处理。空间范围使用影像的空间范围。
2.  使用geoai-py库制作实现数据下载，数据处理（包含裁剪、对齐）、切片、生成样本

   <br />

