from resize_dataset.utils import ConfigDict

from .coco import COCO_TASKS

DATASET_REGISTRY = ConfigDict(coco=COCO_TASKS)

__all__ = ("DATASET_REGISTRY",)
