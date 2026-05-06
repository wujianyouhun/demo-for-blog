import geoai
import sys

def download_naip():
    """
    下载 NAIP 影像
    """
    # 定义边界框（旧金山区域）
    bbox = [-122.51, 37.71, -122.41, 37.81]

    # 下载 2020 年的 NAIP 影像
    downloaded_files = geoai.download_naip(
        bbox=bbox,
        output_dir="naip_data",
        year=2020,
        max_items=5,
        overwrite=False,
        preview=True
    )

    print(f"下载了 {len(downloaded_files)} 个文件:")
    for file in downloaded_files:
        print(f"- {file}")

def download_overture():
    """
    下载 Overture 建筑数据
    """
    # 定义边界框（西安区域）
    bbox =  [107.6, 33.7, 109.8, 34.8]

    # 下载建筑物数据
    try:
        output_file = geoai.download_overture_buildings(
            bbox=bbox,
            output="mxian_buildings.geojson"
        )
        print(f"建筑物数据已保存到: {output_file}")
        
        # 提取建筑物统计信息
        stats = geoai.extract_building_stats(output_file)
        print("建筑物统计信息:")
        print(f"- 总建筑物数: {stats['total_buildings']}")
        print(f"- 有高度信息的建筑物: {stats['has_height']}")
        print(f"- 有名称信息的建筑物: {stats['has_name']}")
    except ImportError as e:
        print(f"需要安装 overturemaps 包: {e}")     

def download_sentinel2():
    """
    下载 Sentinel-2 数据
    """
    # 搜索 Sentinel-2 数据
    items = geoai.pc_stac_search(
        collection="sentinel-2-l2a",
        bbox=[107.6, 33.7, 109.8, 34.8],
        time_range="2024-05-01/2024-05-31",
        max_items=3
    )

    print(f"找到 {len(items)} 个 Sentinel-2 项目")

    # 下载这些项目的 RGB 波段
    if items:
        downloaded = geoai.pc_stac_download(
            items=items,
            output_dir="sentinel2_data",
            assets=["B02", "B03", "B04"],  # 蓝、绿、红波段
            max_workers=3
        )
        
        for item_id, assets in downloaded.items():
            print(f"项目 {item_id} 下载的资产:")
            for asset, path in assets.items():
                print(f"- {asset}: {path}")

def view_sentinel2():
    """
    查看 Sentinel-2 数据
    """
    try:
        # 搜索 Sentinel-2 数据
        items = geoai.pc_stac_search(
            collection="sentinel-2-l2a",
            bbox=[107.6, 33.7, 109.8, 34.8],
            time_range="2024-05-01/2024-05-31",
            max_items=1
        )

        if not items:
            print("没有找到 Sentinel-2 数据")
            return

        print(f"找到项目: {items[0].id}")
        print(f"时间: {items[0].datetime}")
        print(f"可用资产: {list(items[0].assets.keys())}")

        # 尝试可视化
        try:
            m = geoai.view_pc_item(
                item=items[0],
                assets=["visual"],  # 真彩色波段
                name="Sentinel-2 影像",
                basemap="ESRI"
            )
            print("地图可视化成功！")
            # 在 Jupyter Notebook 中显示地图
            try:
                from IPython.display import display
                display(m)
            except ImportError:
                # 如果不是在 Jupyter 环境中，尝试保存为 HTML
                try:
                    output_html = "sentinel2_map.html"
                    m.to_html(output_html)
                    print(f"地图已保存为: {output_html}")
                except Exception as save_error:
                    print(f"无法保存地图: {save_error}")
            return m

        except Exception as visual_error:
            print(f"可视化失败: {visual_error}")
            print("正在尝试替代方案...")
            
            # 替代方案：下载数据后本地查看
            print("建议先下载数据，然后使用本地工具查看")
            download_choice = input("是否要下载数据？(y/n): ").strip().lower()
            if download_choice == 'y':
                download_sentinel2()
            else:
                print("已跳过下载")
                
            return None

    except Exception as e:
        print(f"view_sentinel2 执行失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. Planetary Computer Titiler 服务暂时不可用")
        print("3. 边界框参数不正确")
        print("\n建议解决方案:")
        print("1. 检查网络连接")
        print("2. 稍后重试")
        print("3. 尝试使用下载功能获取数据后本地查看")
        return None

def main():
    """主函数，支持选择下载内容"""
    # 显示欢迎信息和菜单
    print("=" * 60)
    print("GeoAI 数据下载工具")
    print("=" * 60)
    print("\n请选择要下载的数据类型:")
    print("1. NAIP 影像 (美国国家农业影像计划)")
    print("2. Overture 建筑物数据")
    print("3. Sentinel-2 数据 (STAC)")
    print("4. 预览 Sentinel-2 数据 (STAC)")
    print("5. 退出")
    print("-" * 60)

    # 获取用户选择
    choice = input("请输入选项 (1-5): ").strip()
    if choice == "1":
        download_naip()
    elif choice == "2":
        download_overture()
    elif choice == "3":
        download_sentinel2()
    elif choice == "4":
        view_sentinel2()
    elif choice == "5":
        print("谢谢使用！")
        sys.exit(0)
    else:
        print("无效选项，请输入 1-5 之间的数字")

if __name__ == "__main__":
    main()
