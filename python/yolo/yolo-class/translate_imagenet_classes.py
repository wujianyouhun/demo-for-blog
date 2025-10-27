import csv
import time
from openai import OpenAI

# 初始化 OpenAI 客户端（请替换成你自己的 API Key）
client = OpenAI(api_key="you api key")

# 输入的类别英文名文件（每行一个英文类别）
input_file = "yolov8n-cls.txt"

# 输出的 CSV 文件
output_file = "imagenet_classes.csv"

def translate_to_chinese(text):
    """调用 GPT 翻译英文类别名为中文"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 翻译用轻量模型即可
            messages=[
                {"role": "system", "content": "你是一个专业的英中翻译助手。只返回简洁的中文名，不要解释。"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("翻译错误:", e)
        return ""

def main():
    with open(input_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["English Name", "Chinese Translation"])

        for i, name in enumerate(class_names, start=1):
            chinese = translate_to_chinese(name)
            writer.writerow([name, chinese])
            print(f"{i}/{len(class_names)}: {name} -> {chinese}")
            time.sleep(1)  # 避免触发速率限制，可调整

    print(f"\n✅ 已生成翻译文件：{output_file}")

if __name__ == "__main__":
    main()
