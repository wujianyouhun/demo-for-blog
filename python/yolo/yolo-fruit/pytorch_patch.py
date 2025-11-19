
import torch
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel, ultralytics.nn.tasks.SegmentationModel])
