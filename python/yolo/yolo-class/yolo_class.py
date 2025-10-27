"""
YOLO v8 模型信息查看与保存程序 (yolov8n.pt)
=========================================

本程序显示并保存 YOLO v8n 模型的详细信息，包括：
- 所有检测类别列表
- 模型框架信息
- 模型架构信息

运行后会生成一个与模型同名的txt文件，包含所有模型信息。

作者: Claude
日期: 2025-10-27
"""

# 导入必要的库
import torch        # PyTorch - 深度学习框架
from ultralytics import YOLO  # YOLO v8 模型库

# ===================================================================
# PyTorch 2.6 兼容性补丁
# ===================================================================
def patch_torch_load():
    """
    PyTorch 2.6 安全加载补丁
    =====================

    问题: PyTorch 2.6 引入了 weights_only=True 的安全特性，
         但 YOLO 模型需要 weights_only=False 才能正常加载。

    解决方案: 创建一个补丁函数，强制所有 YOLO 模型加载时
              使用 weights_only=False。

    注意: 这是一个临时解决方案，仅用于教学演示。
          在生产环境中应使用官方推荐的解决方案。
    """
    # 保存原始的 torch.load 函数
    original_load = torch.load

    def patched_load(*args, **kwargs):
        """
        补丁后的加载函数
        ================

        参数:
            *args: 传递给原始 torch.load 的位置参数
            **kwargs: 传递给原始 torch.load 的关键字参数

        功能:
            强制设置 weights_only=False，确保 YOLO 模型能正常加载
        """
        # 无论用户如何设置，都强制使用 weights_only=False
        kwargs['weights_only'] = False

        # 调用原始的加载函数
        return original_load(*args, **kwargs)

    # 替换原始的 torch.load 函数
    torch.load = patched_load

# 应用补丁
patch_torch_load()

# ===================================================================
# 获取模型信息函数
# ===================================================================
def get_model_info(model):
    """
    获取 YOLO v8n 模型的详细信息
    ===========================

    参数:
        model: YOLO 模型对象

    返回:
        str: 格式化的模型信息字符串
    """
    model_info = []

    model_info.append("=" * 60)
    model_info.append("YOLO v8n 模型详细信息")
    model_info.append("=" * 60)
    model_info.append("")

    # 模型基本信息
    model_info.append("📋 模型基本信息")
    model_info.append("-" * 30)
    model_info.append(f"模型类型: {model.__class__.__name__}")
    model_info.append(f"模型文件: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'yolov8n.pt'}")
    model_info.append(f"检测类别数量: {len(model.names)}")
    model_info.append("")

    # 显示所有检测类别
    model_info.append("🎯 所有检测类别")
    model_info.append("-" * 30)
    for class_id, class_name in model.names.items():
        model_info.append(f"ID {class_id:2d}: {class_name}")
    model_info.append("")

    # 模型框架信息
    model_info.append("🏗️  模型框架信息")
    model_info.append("-" * 30)
    model_info.append("框架版本: ultralytics YOLO v8")
    model_info.append("深度学习框架: PyTorch")
    model_info.append("任务类型: 目标检测 (Object Detection)")
    model_info.append("模型变体: YOLO v8n (nano - 最小最快)")
    model_info.append("")

    # 输出层信息
    model_info.append("🔧 模型架构信息")
    model_info.append("-" * 30)
    if hasattr(model, 'model') and hasattr(model.model, 'names'):
        model_info.append(f"输出维度: {len(model.names)} 个类别")

    # 打印模型参数统计
    if hasattr(model, 'info'):
        try:
            model_info_data = model.info()
            if isinstance(model_info_data, dict):
                if 'parameters' in model_info_data:
                    model_info.append(f"模型参数: {model_info_data['parameters']:,}")
                if 'GFLOPs' in model_info_data:
                    model_info.append(f"计算量: {model_info_data['GFLOPs']:.1f} GFLOPs")
        except:
            model_info.append("模型统计信息获取失败")

    return '\n'.join(model_info)

# ===================================================================
# 保存模型信息到文件函数
# ===================================================================
def save_model_info_to_file(model, filename):
    """
    保存模型信息到文件
    =================

    参数:
        model: YOLO 模型对象
        filename: 输出文件名
    """
    # 获取模型信息
    model_info_text = get_model_info(model)

    # 添加文件头部信息
    header = []
    header.append("YOLO v8n 模型信息文件")
    header.append("=" * 50)
    header.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.append("")

    full_content = '\n'.join(header) + model_info_text

    # 写入文件
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"✓ 模型信息已保存至: {filename}")
        return True
    except Exception as e:
        print(f"✗ 保存文件失败: {e}")
        return False

# ===================================================================
# 显示模型信息函数
# ===================================================================
def print_model_info(model):
    """
    显示 YOLO v8n 模型的详细信息
    ===========================

    参数:
        model: YOLO 模型对象

    显示内容:
        - 模型基本信息
        - 检测类别列表
        - 模型框架信息
    """
    # 获取并显示模型信息
    model_info_text = get_model_info(model)
    print(model_info_text)

# ===================================================================
# 主函数
# ===================================================================
def main():
    """
    主函数：YOLO v8n 模型信息显示和保存
    ==================================

    本函数显示并保存 YOLO v8n 模型的详细信息：
    1. 加载 YOLO v8n 模型
    2. 显示模型详细信息（类别、框架、架构等）
    3. 保存模型信息到文本文件
    """

    # 1. 加载 YOLO v8n 模型
    print("=" * 50)
    print("正在加载 YOLO v8n 模型...")
    model = YOLO('yolov8n-pose.pt')  # 加载 nano 模型（最小最快）
    print("✓ 模型加载成功")

    # 2. 显示模型详细信息
    print_model_info(model)

    # 3. 保存模型信息到文件
    print("\n" + "=" * 50)
    print("正在保存模型信息...")

    # 根据模型文件名生成输出文件名
    model_filename = 'yolov8n-pose'
    output_filename = f"{model_filename}_model_info.txt"

    if save_model_info_to_file(model, output_filename):
        print("✓ 模型信息保存成功")
    else:
        print("✗ 模型信息保存失败")

    # 4. 程序结束提示
    print("\n" + "=" * 50)
    print("✓ YOLO v8n 模型信息处理完成")
    print("=" * 50)
    print(f"📁 模型信息已保存到: {output_filename}")
    print("\n💡 提示：")
    print("   如果要进行目标检测，请：")
    print("   1. 将图片文件放在当前目录下")
    print("   2. 或者使用其他YOLO检测程序")
    print("=" * 50)

# ===================================================================
# 程序入口
# ===================================================================
if __name__ == "__main__":
    main()