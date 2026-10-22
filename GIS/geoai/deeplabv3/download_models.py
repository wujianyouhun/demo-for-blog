"""下载并校验官方预训练权重到 models/deeplabv3。"""
from common import MODEL_DIR, pretrained_model


if __name__ == "__main__":
    pretrained_model()
    print(f"DeepLabV3-ResNet50 官方权重已就绪：{MODEL_DIR}")
