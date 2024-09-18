# RESIZE DATASET

This repository provides tools to resize computer vision datasets, enabling the enhancement of image and annotation resolutions (such as segmentation masks) using super-resolution techniques and mask refinement.

## ⚠️ Disclaimer

This repository is under continuous development.

## 📋 Features

- [ ] **Dataset Resizing**: Resize the images and labels of the dataset to the specified shape, or scale them by the specified factor.

## Supported Dataset Formats

- [x] COCO format

## ⚙️ Installation

To install **resize-dataset** you can clone the repository and use pip.

1. Clone the repository.

   ```ssh
   git clone https://javierganan99/resize-dataset.git
   cd resize-dataset
   ```

2. Install the tool using pip.

- Just use it (not recommended).

  ```ssh
  pip install .
  ```

- Editable mode.

  ```ssh
  pip install -e .
  ```

## 🖥️ Usage

**resize-dataset** can be accessed through both the Command-Line Interface (CLI) and Python code. The default parameters are configured in the `resize-dataset/cfg/default.yaml` file, and overwritten by the specified arguments in the CLI or Python calls.

### CLI

**resize-dataset** may be used directly in the Command Line Interface (CLI), with the following command format:

```ssh
resize-dataset <task> <arg1=value2> <arg2=value2> ...
```

For example:

```ssh
resize-dataset scale scale_factor=4 dataset_format=coco dataset_task=segmentation show
```

### Python

DAM may also be used directly in a Python environment, and it accepts the same arguments as in the CLI example above:

```python
from resize_dataset import resize_dataset

# Scale a dataset
resize_dataset(task="scale", images_path="/your_path/coco_dataset/val2017", labels_path="/your_path/coco_dataset/panoptic_val2017.json", dataset_format="coco", dataset_task="panoptic")
```
