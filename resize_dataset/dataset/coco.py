import os
from pathlib import Path
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_utils
from resize_dataset.image import RESIZE_METHODS
from resize_dataset.label import (
    VISUALIZATION_REGISTRY,
    rle_to_mask,
    mask_to_rle,
)
from resize_dataset.utils import (
    ConfigDict,
    load_json,
    save_json,
    ensure_folder_exist,
    LOGGER,
)
from resize_dataset.label.conversion import (
    rle_coded_to_binary_mask,
    binary_mask_to_rle_coded,
)
from .utils import generate_n_unique_colors
from .base import ResizableDataset

COCO_TASKS = ConfigDict()


@COCO_TASKS.register(name="segmentation")
@COCO_TASKS.register(name="detection")
class COCODataset(ResizableDataset):
    """
    Dataset class for loading and processing images and annotations in COCO format.

    This class provides functionality to initialize a dataset from a specified images path
    and annotations path, retrieve images and annotations, scale images and annotations based
    on a given factor, and visualize the annotations on images. It is designed for detection
    and segmentation tasks.

    Args:
        images_path (str): Path to the folder containing images.
        annotations_path (str): Path to the COCO annotations JSON file.
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.

    Attributes:
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.
        images_folder (str): Path to the folder containing images.
        annotations (COCO): COCO object representing the annotations.
        ids (list): Sorted list of image IDs in the annotations.
        id2name (dict): Mapping from category ID to category name.
        name2id (dict): Mapping from category name to category ID.
        id2color (dict): Mapping from category ID to a unique color.

    Methods:
        scale(img, anns, scale_factor, resize_image_method="bicubic"): Scales the image and
            annotations by the given scale factor.
        show(image, anns): Displays the image with its corresponding annotations.
        __getitem__(index): Retrieves the image and annotations for the given index.
        __len__(): Returns the total number of images in the dataset.

    Private Methods:
        _create_annotations(annotations_path): Initializes the annotations from the COCO format
            JSON file and sets up internal mappings.
        save(index, image, anns): Saves an image and its annotations to the output folder
            and dictionary.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        self.ids = self._get_valid_image_ids()
        self.id2name = {k: v["name"] for k, v in self.annotations.cats.items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        # To save
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        self.output_annotations = self.annotations.dataset
        self.output_annotations["annotations"] = []
        self.output_annotations["images"] = []
        if self.cfg.save:
            ensure_folder_exist(self.images_output_folder)
        # To show
        self._window_name = "Annotations"
        self._window = None

    def _create_annotations(self, annotations_path):
        """
        Creates and organizes annotations from a specified JSON file.

        This function loads annotations from a given path and structures them into
        a dictionary-like object. It organizes the annotations by mapping image IDs
        to their corresponding annotations, as well as organizing categories and
        images. The resulting structure is stored in the `annotations` attribute
        of the class.

        Args:
            annotations_path (str): The file path to the JSON annotations file.

        Returns:
            None: This function does not return a value. It modifies the
            annotations attribute of the class instance in place.
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["anns"], self.annotations["cats"], self.annotations["imgs"] = (
            {},
            {},
            {},
        )
        self.annotations["imgToAnns"], self.annotations["catToImgs"] = (
            ConfigDict(),
            ConfigDict(),
        )
        if "annotations" in self.annotations.dataset:
            for ann in self.annotations.dataset["annotations"]:
                if ann["image_id"] not in self.annotations.imgToAnns:
                    self.annotations.imgToAnns[ann["image_id"]] = []
                self.annotations.imgToAnns[ann["image_id"]].append(ann)
                self.annotations["anns"][ann["id"]] = ann
        if "images" in self.annotations.dataset:
            for img in self.annotations.dataset["images"]:
                self.annotations["imgs"][img["id"]] = img
        if "categories" in self.annotations.dataset:
            for cat in self.annotations.dataset["categories"]:
                self.annotations["cats"][cat["id"]] = cat

    def _get_valid_image_ids(self):
        """
        Checks which images can be successfully loaded and returns a list of valid image IDs.

        Returns:
            list: A list of valid image IDs.
        """
        valid_ids = []
        for img_id, img_data in self.annotations["imgs"].items():
            img_path = str(Path(self.images_folder) / img_data["file_name"])
            if not os.path.exists(img_path):
                LOGGER.info(
                    "Can't open filepath %s, it could not exist or be corrupted.",
                    img_path,
                )
                continue
            valid_ids.append(img_id)
        return sorted(valid_ids)

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales an image and its corresponding annotations by a specified factor.

        This function resizes the input image using the specified resizing method
        and scales the bounding boxes, segmentation masks, and area of the
        annotations accordingly. It handles both polygon and RLE segmentation formats.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (list of dict): The annotations containing bounding boxes and
                segmentation data to be scaled.
            scale_factor (float): The factor by which to scale the image and annotations.
            resize_image_method (str, optional): The method to use for resizing the image
                (default is "bicubic").

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and the modified
                annotations (list of dict).
        """
        img = RESIZE_METHODS.get(resize_image_method)(img, scale_factor)
        for ann in anns:
            ann["bbox"] = [c * scale_factor for c in ann["bbox"]]
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):  # Segmentation polygon
                    for i, polygon in enumerate(ann["segmentation"]):
                        ann["segmentation"][i] = [c * scale_factor for c in polygon]
                else:
                    seg = ann["segmentation"]
                    h, w = seg["size"]
                    if isinstance(ann["segmentation"]["counts"], str):
                        mask = mask_utils.decode(ann["segmentation"])
                    else:
                        mask = rle_to_mask(seg["counts"], h, w, order="F")
                    resized_mask = cv2.resize(
                        mask,
                        (w * scale_factor, h * scale_factor),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    if isinstance(ann["segmentation"]["counts"], str):
                        ann["segmentation"]["counts"] = binary_mask_to_rle_coded(
                            resized_mask
                        )
                    else:
                        ann["segmentation"]["counts"] = mask_to_rle(
                            resized_mask, order="F"
                        )
                    ann["segmentation"]["size"] = [
                        h * scale_factor,
                        w * scale_factor,
                    ]
            ann["area"] *= scale_factor**2
        return img, anns

    def reshape(self, img, anns, shape, resize_image_method="bicubic"):
        """
        Reshapes an image and its corresponding annotations to a specified size.

        This function resizes the input image and scales the bounding boxes and
        segmentation annotations according to the new dimensions. It offers the
        option to specify the method used for resizing the image.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (list[dict]): A list of annotations corresponding to the image,
                each containing at least a "bbox" key and optionally "segmentation".
            shape (tuple): The desired output shape of the image as (width, height).
            resize_image_method (str, optional): The method used for resizing the
                image, default is "bicubic". Other methods can be defined in
                RESIZE_METHODS.

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and the updated
            annotations (list[dict]).
        """
        h0, w0 = img.shape[:2]
        wn, hn = shape
        xf = wn / w0
        yf = hn / h0
        img = RESIZE_METHODS.get(resize_image_method)(img, shape)
        for ann in anns:
            x1, y1, w, h = ann["bbox"]
            ann["bbox"] = [x1 * xf, y1 * yf, w * xf, h * yf]
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):  # Segmentation polygon
                    for i, polygon in enumerate(ann["segmentation"]):
                        ann["segmentation"][i] = [
                            c * xf if i % 2 == 0 else c * yf
                            for i, c in enumerate(polygon)
                        ]
                else:
                    seg = ann["segmentation"]
                    h0, w0 = seg["size"]
                    xf = wn / w0
                    yf = hn / h0
                    if isinstance(ann["segmentation"]["counts"], str):
                        mask = mask_utils.decode(ann["segmentation"])
                    else:
                        mask = rle_to_mask(seg["counts"], h0, w0, order="F")
                    resized_mask = cv2.resize(
                        mask, shape, interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)
                    if isinstance(ann["segmentation"]["counts"], str):
                        ann["segmentation"]["counts"] = binary_mask_to_rle_coded(
                            resized_mask
                        )
                    else:
                        ann["segmentation"]["counts"] = mask_to_rle(
                            resized_mask, order="F"
                        )
                    ann["segmentation"]["size"] = [hn, wn]
            ann["area"] *= xf * yf
        return img, anns

    def save(self, index, image, anns):
        """
        Saves an image and its annotations to the output folder and dictionary.

        This function writes the image to the images output folder and appends
        the annotations to the output_annotations dictionary.

        Args:
            index (int): The index of the image in the dataset.
            image (np.ndarray): The processed image to be saved.
            anns (list): The annotations to be saved.

        Returns:
            None: This function does not return a value.
        """
        img_id = self.ids[index]
        img_ann = self.annotations.imgs[img_id]
        cv2.imwrite(str(Path(self.images_output_folder) / img_ann["file_name"]), image)
        img_ann["height"], img_ann["width"] = image.shape[:2]
        self.output_annotations["annotations"].extend(anns)
        self.output_annotations["images"].append(img_ann)

    def show(self, image, anns):
        """
        Displays an image with annotations such as segmentation masks, bounding boxes, or polygons.

        This function takes an image represented as a NumPy array and a list of annotation
        dictionaries, visualizing the annotations on the image. Annotations may include
        segmentation data or bounding box data.
        The modified image is displayed in a window until a key is pressed.

        Args:
            image (np.ndarray): The image to annotate, represented as a NumPy array.
            anns (list): A list of annotation dictionaries, where each dictionary can contain
                segmentation data or bounding box data with category id.

        Returns:
            None: This function does not return any value.
        """
        img_with_annotations = image.copy()
        for ann in anns:
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):  # Segmentation polygon
                    label = ann["segmentation"]
                    visualize = VISUALIZATION_REGISTRY.polygons
                else:
                    seg = ann["segmentation"]
                    h, w = seg["size"]
                    if (
                        isinstance(seg["counts"], list)
                        and len(seg["counts"]) == 1
                        and isinstance(seg["counts"][0], str)
                    ):
                        seg["counts"] = seg["counts"][0]
                        seg["size"] = [
                            seg["size"][0].item(),
                            seg["size"][1].item(),
                        ]
                    if isinstance(seg["counts"], str):
                        label = rle_coded_to_binary_mask(seg["counts"], h, w)
                    else:
                        label = rle_to_mask(seg["counts"], h, w, order="F")
                    visualize = VISUALIZATION_REGISTRY.mask
            else:
                label = ann["bbox"]
                visualize = VISUALIZATION_REGISTRY.bbox
            class_id = (
                ann["category_id"].item()
                if isinstance(ann["category_id"], torch.Tensor)
                else ann["category_id"]
            )
            visualize(
                img_with_annotations,
                label,
                self.id2name[class_id],
                color=self.id2color[class_id],
            )
        if not self._window:
            self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self._window_name, img_with_annotations)
        cv2.waitKey(0)

    def __getitem__(self, index):
        """
        Retrieves an image and its associated annotations from the dataset based on the given index.

        This function accesses the dataset using the specified index to return the processed image
        and its corresponding annotations. It checks for available annotations related to the image
        and applies scaling before returning the results.

        Args:
            index (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the processed image (np.ndarray) and a list of annotations.
        """
        img_id = self.ids[index]
        ann_ids = (
            [ann["id"] for ann in self.annotations.imgToAnns[img_id]]
            if img_id in self.annotations.imgToAnns
            else []
        )
        anns = [self.annotations.anns[idx] for idx in ann_ids]
        path = self.annotations.imgs[img_id]["file_name"]
        img = cv2.imread(str(Path(self.images_folder) / path))
        resize_function, resize_parameter = (
            (self.scale, self.cfg.scale_factor)
            if self.cfg.image_shape is None
            else (self.reshape, self.cfg.image_shape)
        )
        img, anns = resize_function(
            img,
            anns,
            resize_parameter,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        if self.cfg.show:
            self.show(img, anns)
        return img, anns

    def __len__(self):
        """
        Returns the number of images in the dataset.

        This function calculates and returns the total number of images available in
        the dataset by determining the length of the internal list of image identifiers.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.ids)

    def close(self):
        """
        Saves all annotations to a JSON file specified in the labels_output_path.

        This function writes the output_annotations dictionary to a JSON file
        in the specified labels output path.
        """
        save_json(self.output_annotations, self.labels_output_path)
        if self._window:
            cv2.destroyWindow(self._window_name)


@COCO_TASKS.register(name="panoptic")
class COCODatasetPanoptic(ResizableDataset):
    """
    Dataset class for loading and processing images and annotations in COCO format
    for Panoptic Segmentation tasks (https://cocodataset.org/#format-data).

    This class provides functionality to initialize a dataset from specified image and
    annotation paths, retrieve images and annotations, scale images and annotations based
    on a given factor, and visualize the annotations on images.

    Args:
        images_path (str): Path to the folder containing images.
        annotations_path (str): Path to the COCO annotations JSON file.
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.

    Attributes:
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.
        images_folder (str): Path to the folder containing images.
        annotations (COCO): COCO object representing the annotations.
        ids (list): Sorted list of image IDs in the annotations.
        id2name (dict): Mapping from category ID to category name.
        name2id (dict): Mapping from category name to category ID.
        id2color (dict): Mapping from category ID to a unique color.
        images_output_folder (str): Path to save the output processed images.
        labels_output_folder (str): Path to save the output labels.
        labels_output_path (str): Path to the output JSON file for annotations.
        output_annotations (dict): Dictionary to store output annotations for saving.

    Methods:
        __init__(images_path, annotations_path, cfg): Initializes the dataset with
            images and annotations, and prepares the necessary mappings.

        scale(img, anns, scale_factor, resize_image_method="bicubic"): Scales an
            image and its corresponding annotations by a specified factor.

        save(index, image, anns): Saves an image and its annotations to the output
            folders and dictionary.

        show(image, anns): Displays an image with its annotations such as segmentation
            masks, bounding boxes, or polygons.

        __getitem__(index): Retrieves an image and its associated annotations from
            the dataset based on the given index.

        __len__(): Returns the number of images in the dataset.

        close(): Saves all annotations to a JSON file specified in the labels_output_path.

    Private Methods:
        _create_annotations(annotations_path): Creates and parses the annotations
            from the COCO format based on the provided annotations path.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        self.annotations_folder = Path(annotations_path).with_suffix("")
        self.ids = self._get_valid_image_ids()
        self.id2name = {k: v["name"] for k, v in self.annotations.cats.items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        # To save
        self.images_output_folder = self.cfg.images_output_path
        if self.cfg.labels_output_path.endswith(".json"):
            self.labels_output_folder = self.cfg.labels_output_path[:-5]
            self.labels_output_path = self.cfg.labels_output_path
        else:
            self.labels_output_folder = self.cfg.labels_output_path
            self.labels_output_path = self.cfg.labels_output_path + ".json"
        self.output_annotations = self.annotations.dataset
        self.output_annotations["annotations"] = []
        self.output_annotations["images"] = []
        if self.cfg.save:
            ensure_folder_exist(self.images_output_folder)
        # To show
        self._window_name = "Annotations"
        self._window = None

    def _create_annotations(self, annotations_path):
        """
        Creates and initializes annotations for a dataset from a JSON file.

        This function loads the annotations from a specified JSON file and populates
        the internal data structures for annotations, images, and categories. It
        checks for the presence of "annotations", "images", and "categories" in the
        loaded dataset and organizes the data accordingly.

        Args:
            annotations_path (str): The file path to the JSON annotations file.

        Returns:
            None: This function does not return a value but initializes the
            'annotations' attribute of the class instance.
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["anns"], self.annotations["cats"], self.annotations["imgs"] = (
            {},
            {},
            {},
        )
        if "annotations" in self.annotations.dataset:
            for ann in self.annotations.dataset["annotations"]:
                self.annotations["anns"][ann["image_id"]] = ann
        if "images" in self.annotations.dataset:
            for img in self.annotations.dataset["images"]:
                self.annotations["imgs"][img["id"]] = img
        if "categories" in self.annotations.dataset:
            for cat in self.annotations.dataset["categories"]:
                self.annotations["cats"][cat["id"]] = cat

    def _get_valid_image_ids(self):
        """
        Checks which images can be successfully loaded and returns a list of valid image IDs.

        Returns:
            list: A list of valid image IDs.
        """
        valid_ids = []
        for img_id, img_data in self.annotations["imgs"].items():
            img_path = str(Path(self.images_folder) / img_data["file_name"])
            if not os.path.exists(img_path):
                LOGGER.info(
                    "Can't open filepath %s, it could not exist or be corrupted.",
                    img_path,
                )
                continue
            valid_ids.append(img_id)
        return sorted(valid_ids)

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales an image and its corresponding annotations by a specified factor.

        This function resizes the input image using the specified resizing method
        and scales the segmentation masks, bounding boxes, and area of the
        annotations accordingly.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (dict): The annotations containing segmentation data to be scaled.
            scale_factor (float): The factor by which to scale the image and annotations.
            resize_image_method (str, optional): The method to use for resizing the image
                (default is "bicubic").

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and a tuple
                containing the modified annotations (dict) and scaled segmentation mask
                    (np.ndarray).
        """
        img = RESIZE_METHODS.get(resize_image_method)(img, scale_factor)
        seg_path = str(self.annotations_folder / anns["file_name"])
        seg_img = cv2.imread(seg_path, cv2.IMREAD_COLOR)
        scaled_segmentation = cv2.resize(
            seg_img,
            (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
        for seg_info in anns["segments_info"]:
            seg_info["bbox"] = [c * scale_factor for c in seg_info["bbox"]]
            seg_info["area"] *= scale_factor**2
        return img, (anns, scaled_segmentation)

    def reshape(self, img, anns, shape, resize_image_method="bicubic"):
        """
        Reshapes an image and its corresponding segmentation annotations to a specified size.

        This function resizes the input image using a specified interpolation method
        and adjusts the corresponding segmentation annotations to match the new dimensions.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (dict): The annotations corresponding to the image, including
                segmentation information.
            shape (tuple): The desired shape for the output image as (width, height).
            resize_image_method (str, optional): The method to use for resizing the image
                (default is "bicubic").

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and a tuple of updated
                annotations (dict) along with the resized segmentation mask (np.ndarray).
        """
        h0, w0 = img.shape[:2]
        wn, hn = shape
        xf = wn / w0
        yf = hn / h0
        img = RESIZE_METHODS.get(resize_image_method)(img, shape)
        seg_path = str(self.annotations_folder / anns["file_name"])
        seg_img = cv2.imread(seg_path, cv2.IMREAD_COLOR)
        scaled_segmentation = cv2.resize(
            seg_img,
            shape,
            interpolation=cv2.INTER_NEAREST,
        )
        for seg_info in anns["segments_info"]:
            x1, y1, w, h = seg_info["bbox"]
            seg_info["bbox"] = [x1 * xf, y1 * yf, w * xf, h * yf]
            seg_info["area"] *= xf * yf
        return img, (anns, scaled_segmentation)

    def save(self, index, image, anns):
        """
        Saves an image and its annotations to the output folders and dictionary.

        This function writes the image and annotation to the specified output folders
        and appends the annotations in the output_annotations dictionary. The function
        handles both the image file and its corresponding mask for annotations.

        Args:
            index (int): The index of the image in the dataset.
            image (np.ndarray): The processed image to be saved.
            anns (list): The annotations to be saved, where the first element is the
                annotation dictionary and the second element is the scaled mask.

        Returns:
            None: This function does not return a value.
        """
        annotation, scaled_mask = anns
        img_id = self.ids[index]
        img_ann = self.annotations.imgs[img_id]
        cv2.imwrite(str(Path(self.images_output_folder) / img_ann["file_name"]), image)
        cv2.imwrite(
            str(Path(self.labels_output_folder) / annotation["file_name"]),
            scaled_mask,
        )
        img_ann["height"], img_ann["width"] = image.shape[:2]
        self.output_annotations["annotations"].append(annotation)
        self.output_annotations["images"].append(img_ann)

    def show(self, image, anns):
        """
        Displays an image with annotations such as segmentation masks, bounding boxes, or polygons.

        This function takes an image and its annotations, and visualizes the annotations
        on the image. Annotations may include segmentation information or bounding box data.
        The modified image is displayed in a window until a key is pressed.

        Args:
            image (np.ndarray): The image to annotate, represented as a NumPy array.
            anns (tuple): A tuple containing:
                - annotation (dict): The annotation dictionary containing segmentation and
                    bounding box data.
                - scaled_mask (np.ndarray): The scaled mask to assist in visualization of segments.

        Returns:
            None: This function does not return any value.
        """
        annotation, scaled_mask = anns
        img_with_annotations = image.copy()
        seg_img = scaled_mask.astype(np.uint32)
        ids = seg_img[:, :, 2] + seg_img[:, :, 1] * 256 + seg_img[:, :, 0] * 256**2
        for seg_info in annotation["segments_info"]:
            idx = (
                seg_info["id"].item()
                if isinstance(seg_info["id"], torch.Tensor)
                else seg_info["id"]
            )
            class_id = (
                seg_info["category_id"].item()
                if isinstance(seg_info["category_id"], torch.Tensor)
                else seg_info["category_id"]
            )
            mask = (ids == idx).astype(np.uint8)
            color = self.id2color[class_id]
            VISUALIZATION_REGISTRY.mask(
                img_with_annotations,
                mask,
                self.id2name[class_id],
                color=color,
            )
        if not self._window:
            self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self._window_name, img_with_annotations)
        cv2.waitKey(0)

    def __getitem__(self, index):
        """
        Retrieves an image and its associated annotations from the dataset based on the given index.

        This function allows you to access images and their corresponding annotations within
        the dataset. It reads the image file from the disk, processes it, and returns both the
        image and its annotations.

        Args:
            index (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the processed image (np.ndarray) and the annotations
                (list of dicts).
        """
        img_id = self.ids[index]
        img_info = self.annotations.imgs[img_id]
        img = cv2.imread(str(Path(self.images_folder) / img_info["file_name"]))
        anns = self.annotations.anns[img_id]
        resize_function, resize_parameter = (
            (self.scale, self.cfg.scale_factor)
            if self.cfg.image_shape is None
            else (self.reshape, self.cfg.image_shape)
        )
        img, anns = resize_function(
            img,
            anns,
            resize_parameter,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        if self.cfg.show:
            self.show(img, anns)
        return img, anns

    def __len__(self):
        """
        Returns the number of images in the dataset.

        This method calculates and returns the total count of images stored in
        the dataset by measuring the length of the ids attribute, which is expected
        to hold the identifiers for each image.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.ids)

    def close(self):
        """
        Saves all annotations to a JSON file specified in the labels_output_path.

        This function writes the output_annotations dictionary to a JSON file
        in the specified labels output path.
        """
        save_json(
            self.output_annotations,
            self.labels_output_path,
        )
        if self._window:
            cv2.destroyWindow(self._window_name)


@COCO_TASKS.register(name="densepose")
class COCODatasetDensePose(ResizableDataset):
    """
    Dataset class for loading and processing images and DensePose annotations in COCO format.

    This class enables the management of a dataset containing images and their associated
    DensePose annotations formatted according to the COCO standard. It provides methods
    for filtering valid images, scaling and reshaping images and annotations, and visualizing
    the results.

    Args:
        images_path (str): Path to the folder containing images.
        annotations_path (str): Path to the COCO JSON annotations file.
        cfg (Config): Configuration object containing various settings.

    Attributes:
        cfg (Config): Configuration settings for the dataset.
        images_folder (str): Path to the images folder.
        ids (list[int]): List of valid image IDs.
        id2name (dict[int, str]): Mapping from image IDs to image names.
        name2id (dict[str, int]): Mapping from image names to image IDs.
        id2color (dict[int, tuple[int]]): Mapping of image IDs to color values.
        images_output_folder (str): Output folder for saving processed images.
        labels_output_path (str): Path to save output annotations.
        output_annotations (dict): Structure containing output annotations data.

    Methods:
        __getitem__(index): Retrieves and processes an image and its associated
            DensePose annotations.
        __len__(): Returns the number of images in the dataset.
        save(index, image, anns): Saves the processed image and its
            corresponding annotations.
        close(): Saves all DensePose annotations to a JSON file specified in the
            labels_output_path.
        scale(img, anns, scale_factor, resize_image_method="bicubic"): Scales an
            image and its associated DensePose annotations.
        reshape(img, anns, shape, resize_image_method="bicubic"): Reshapes an
            image and its corresponding annotations to a specified size.
        show(image, anns): Displays an image with its associated DensePose annotations.

    Private Methods:
        _create_annotations(annotations_path): Creates and organizes annotations from
            a specified JSON file.
        _filter_valid_images(): Checks which images can be successfully loaded and
            returns a list of valid image IDs.
        _filter_valid_annotations(): Updates annotations to include only those related
            to valid images.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        self.ids = self._get_valid_image_ids()
        self.id2name = {k: v["name"] for k, v in self.annotations.cats.items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        self.output_annotations = self.annotations.dataset
        self.output_annotations["annotations"] = []
        self.output_annotations["images"] = []
        if self.cfg.save:
            ensure_folder_exist(self.images_output_folder)
        # To show
        self._window_name = "Annotations"
        self._window = None

    def _create_annotations(self, annotations_path):
        """
        Creates and organizes annotations from a specified JSON file.

        This function loads annotations from a given path and structures them into
        a dictionary-like object. It organizes the annotations by mapping image IDs
        to their corresponding annotations, as well as organizing categories and
        images. The resulting structure is stored in the `annotations` attribute
        of the class.

        Args:
            annotations_path (str): The file path to the JSON annotations file.

        Returns:
            None
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["anns"], self.annotations["cats"], self.annotations["imgs"] = (
            {},
            {},
            {},
        )
        self.annotations["imgToAnns"], self.annotations["catToImgs"] = (
            ConfigDict(),
            ConfigDict(),
        )
        if "annotations" in self.annotations.dataset:
            for ann in self.annotations.dataset["annotations"]:
                if ann["image_id"] not in self.annotations.imgToAnns:
                    self.annotations.imgToAnns[ann["image_id"]] = []
                self.annotations.imgToAnns[ann["image_id"]].append(ann)
                self.annotations["anns"][ann["id"]] = ann
        if "images" in self.annotations.dataset:
            for img in self.annotations.dataset["images"]:
                self.annotations["imgs"][img["id"]] = img
        if "categories" in self.annotations.dataset:
            for cat in self.annotations.dataset["categories"]:
                self.annotations["cats"][cat["id"]] = cat

    def _get_valid_image_ids(self):
        """
        Checks which images can be successfully loaded and returns a list of valid image IDs.

        Returns:
            list: A list of valid image IDs.
        """
        valid_ids = []
        for img_id, img_data in self.annotations["imgs"].items():
            img_path = str(Path(self.images_folder) / img_data["file_name"])
            if not os.path.exists(img_path):
                LOGGER.info(
                    "Can't open filepath %s, it could not exist or be corrupted.",
                    img_path,
                )
                continue
            valid_ids.append(img_id)
        return sorted(valid_ids)

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales an image and its associated DensePose annotations by a specified factor.

        This function rescales the input image and adjusts the DensePose annotations
        (including bounding boxes, masks, keypoints, and segmentations) according to
        the given `scale_factor`. The annotations are modified in place to reflect
        the new scale.

        Args:
            img (np.ndarray): The input image to be scaled.
            anns (list[dict]): A list of annotations, each annotation being a dictionary
                that may contain:
                - "bbox" (list[float]): The bounding box coordinates [x, y, width, height].
                - "dp_masks" (list[dict]): DensePose masks represented as dictionaries containing:
                    - "size" (list[int]): The size of the mask [height, width].
                    - "counts" (str or list[int]): RLE-encoded mask or binary mask counts.
                - "area" (float): The area of the annotated region.
                - "keypoints" (list[float]): Keypoint coordinates and visibility flags
                    [x1, y1, v1, x2, y2, v2, ...].
                - "segmentation" (list[list[float]]): Segmentation polygon coordinates.
            scale_factor (float): The factor by which to scale the image and annotations.
            resize_image_method (str, optional): The interpolation method to use when
                resizing the image. Defaults to "bicubic". Available methods should
                be present in the `RESIZE_METHODS` dictionary.

        Returns:
            tuple:
                - np.ndarray: The scaled image.
                - list[dict]: The scaled annotations with updated bounding boxes, masks,
                    keypoints, segmentations, and area.
        """
        img = RESIZE_METHODS.get(resize_image_method)(img, scale_factor)
        for ann in anns:
            if "bbox" in ann:
                bbox = ann["bbox"]
                ann["bbox"] = [coord * scale_factor for coord in bbox]
            if "dp_masks" in ann:
                for idx, dp_mask in enumerate(ann["dp_masks"]):
                    if isinstance(dp_mask, dict):
                        h, w = dp_mask["size"]
                        if isinstance(dp_mask["counts"], str):
                            mask = mask_utils.decode(dp_mask)
                        else:
                            mask = rle_to_mask(dp_mask["counts"], h, w, order="F")
                        resized_mask = cv2.resize(
                            mask,
                            (w * scale_factor, h * scale_factor),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(np.uint8)
                        if isinstance(dp_mask["counts"], str):
                            ann["dp_masks"][idx]["counts"] = binary_mask_to_rle_coded(
                                resized_mask
                            )
                        else:
                            ann["dp_masks"][idx]["counts"] = mask_to_rle(
                                resized_mask, order="F"
                            )
                        ann["dp_masks"][idx]["size"] = [
                            h * scale_factor,
                            w * scale_factor,
                        ]
            ann["area"] = ann["area"] * (scale_factor**2)
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                scaled_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i] * scale_factor
                    y = keypoints[i + 1] * scale_factor
                    v = keypoints[i + 2]
                    scaled_keypoints.extend([x, y, v])
                ann["keypoints"] = scaled_keypoints
            if "segmentation" in ann:
                ann["segmentation"] = [
                    [coord * scale_factor for coord in segment]
                    for segment in ann["segmentation"]
                ]
        return img, anns

    def reshape(self, img, anns, shape, resize_image_method="bicubic"):
        """
        Reshapes an image and its corresponding annotations to a specified size.

        This function resizes the input image and scales the bounding boxes,
        segmentation and all annotations according to the new dimensions. It offers the
        option to specify the method used for resizing the image.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (list[dict]): A list of annotations corresponding to the image,
                each containing at least a "bbox" key and optionally "segmentation".
            shape (tuple): The desired output shape of the image as (width, height).
            resize_image_method (str, optional): The method used for resizing the
                image, default is "bicubic". Other methods can be defined in
                RESIZE_METHODS.

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and the updated
            annotations (list[dict]).
        """
        h0, w0 = img.shape[:2]
        wn, hn = shape
        xf = wn / w0
        yf = hn / h0
        img = RESIZE_METHODS.get(resize_image_method)(img, shape)
        for ann in anns:
            if "bbox" in ann:
                x1, y1, w, h = ann["bbox"]
                ann["bbox"] = [x1 * xf, y1 * yf, w * xf, h * yf]
            if "dp_masks" in ann:
                for idx, dp_mask in enumerate(ann["dp_masks"]):
                    if isinstance(dp_mask, dict):
                        h, w = dp_mask["size"]
                        if isinstance(dp_mask["counts"], str):
                            mask = mask_utils.decode(dp_mask)
                        else:
                            mask = rle_to_mask(dp_mask["counts"], h, w, order="F")
                        resized_mask = cv2.resize(
                            mask, shape, interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)
                        if isinstance(dp_mask["counts"], str):
                            ann["dp_masks"][idx]["counts"] = binary_mask_to_rle_coded(
                                resized_mask
                            )
                        else:
                            ann["dp_masks"][idx]["counts"] = mask_to_rle(
                                resized_mask, order="F"
                            )
                        ann["dp_masks"][idx]["size"] = [
                            h * yf,
                            w * xf,
                        ]
            ann["area"] *= xf * yf
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                scaled_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i] * xf
                    y = keypoints[i + 1] * yf
                    v = keypoints[i + 2]
                    scaled_keypoints.extend([x, y, v])
                ann["keypoints"] = scaled_keypoints
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):
                    for i, polygon in enumerate(ann["segmentation"]):
                        ann["segmentation"][i] = [
                            c * xf if i % 2 == 0 else c * yf
                            for i, c in enumerate(polygon)
                        ]
        return img, anns

    def show(self, image, anns):
        """
        Displays an image with its associated DensePose annotations.

        This function overlays the DensePose annotations onto a copy of the input image and
        visualizes it using the appropriate visualization method from the `VISUALIZATION_REGISTRY`.
        The displayed image includes features such as DensePose coordinates.

        Args:
            image (np.ndarray): The input image to be displayed.
            anns (list[dict]): A list of annotations associated with the image,
                containing DensePose data such as:
                - "dp_masks": DensePose masks for different body parts.
                - "keypoints": Keypoints for different body regions.
                - Other DensePose-specific annotations.
        """
        if not self._window:
            self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        img_with_annotations = image.copy()
        stacked_annotations = VISUALIZATION_REGISTRY.densepose(
            img_with_annotations, anns
        )
        cv2.imshow(self._window_name, stacked_annotations)
        cv2.waitKey(0)

    def __getitem__(self, index):
        """
        Retrieves and processes an image and its associated DensePose annotations
        based on the given index.

        This function fetches an image and its corresponding annotations from the dataset,
        applies scaling or reshaping according to the configuration settings, and optionally
        saves or displays the processed image and annotations.

        Args:
            index (int): The index of the image and annotations to retrieve.

        Returns:
            tuple:
                - np.ndarray: The processed image.
                - list[list[dict]]: A list containing a single list of annotations
                    associated with the image.

        Workflow:
            1. Retrieves the image ID and its corresponding annotations.
            2. Loads the image from the specified path.
            3. Applies scaling or reshaping based on configuration:
                - Scaling is applied if `self.cfg.image_shape` is `None`.
                - Reshaping is applied if `self.cfg.image_shape` is specified.
            4. If enabled in the configuration (`self.cfg.save`), saves the processed
                image and annotations.
            5. If enabled in the configuration (`self.cfg.show`), displays the image
                with annotations.
        """
        img_id = self.ids[index]
        anns = self.annotations["imgToAnns"].get(img_id, [])
        path = self.annotations["imgs"][img_id]["file_name"]
        img = cv2.imread(str(Path(self.images_folder) / path))
        resize_function, resize_parameter = (
            (self.scale, self.cfg.scale_factor)
            if self.cfg.image_shape is None
            else (self.reshape, self.cfg.image_shape)
        )
        img, anns = resize_function(
            img,
            anns,
            resize_parameter,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        if self.cfg.show:
            self.show(img, anns)
        return img, anns

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.ids)

    def save(self, index, image, anns):
        """
        Saves the processed image and its corresponding annotations.

        This method saves the processed image to the specified output folder, updates
        the image's metadata, and appends the metadata and annotations to the output
        annotations list.

        Args:
            index (int): The index of the image in the dataset, used to retrieve the image ID.
            image (np.ndarray): The processed image to be saved.
            anns (dict): The annotations associated with the image.

        Returns:
            None
        """
        img_id = self.ids[index]
        img_ann = self.annotations.imgs[img_id]
        cv2.imwrite(str(Path(self.images_output_folder) / img_ann["file_name"]), image)
        img_ann["height"], img_ann["width"] = image.shape[:2]
        self.output_annotations["annotations"].append(anns)
        self.output_annotations["images"].append(img_ann)

    def close(self):
        """
        Saves all DensePose annotations to a JSON file specified in the labels_output_path.
        """
        save_json(self.output_annotations, self.labels_output_path)
        if self._window:
            cv2.destroyWindow(self._window_name)


@COCO_TASKS.register(name="caption")
class COCODatasetCaption(ResizableDataset):
    """
    Dataset class for loading and processing images and captions in COCO format.

    This class provides functionality to initialize a dataset from a specified images path
    and annotations path, retrieve images and captions, scale images based on a given factor,
    and visualize the images with captions. It is designed for image captioning tasks.

    Args:
        images_path (str): Path to the folder containing images.
        annotations_path (str): Path to the COCO annotations JSON file.
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.

    Attributes:
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.
        images_folder (str): Path to the folder containing images.
        annotations (COCO): COCO object representing the annotations.
        ids (list): Sorted list of image IDs in the annotations.
        captions (dict): Mapping from image ID to a list of captions.

    Methods:
        scale(img, scale_factor, resize_image_method="bicubic"): Scales the image
            by the given scale factor.
        show(image, caption): Displays the image with its corresponding caption.
        __getitem__(index): Retrieves the image and caption for the given index.
        __len__(): Returns the total number of images in the dataset.

    Private Methods:
        _create_annotations(annotations_path): Initializes the annotations from the
            COCO format JSON file.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        self.ids = self._get_valid_image_ids()
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        self.output_annotations = self.annotations.dataset
        self.output_annotations["annotations"] = []
        self.output_annotations["images"] = []
        if self.cfg.save:
            ensure_folder_exist(self.images_output_folder)
        # To show
        self._window_name = "Annotations"
        self._window = None

    def _create_annotations(self, annotations_path):
        """
        Creates and organizes annotations from a specified JSON file.

        This function loads annotations from a given path and structures them into
        a dictionary-like object. It organizes the annotations by mapping image IDs
        to their corresponding annotations, as well as organizing categories and
        images. The resulting structure is stored in the `annotations` attribute
        of the class.

        Args:
            annotations_path (str): The file path to the JSON annotations file.

        Returns:
            None
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["anns"], self.annotations["cats"], self.annotations["imgs"] = (
            {},
            {},
            {},
        )
        self.annotations["imgToAnns"], self.annotations["catToImgs"] = (
            ConfigDict(),
            ConfigDict(),
        )
        if "annotations" in self.annotations.dataset:
            for ann in self.annotations.dataset["annotations"]:
                if ann["image_id"] not in self.annotations.imgToAnns:
                    self.annotations.imgToAnns[ann["image_id"]] = []
                self.annotations.imgToAnns[ann["image_id"]].append(ann)
                self.annotations["anns"][ann["id"]] = ann
        if "images" in self.annotations.dataset:
            for img in self.annotations.dataset["images"]:
                self.annotations["imgs"][img["id"]] = img

    def _get_valid_image_ids(self):
        """
        Checks which images can be successfully loaded and returns a list of valid image IDs.

        Returns:
            list: A list of valid image IDs.
        """
        valid_ids = []
        for img_id, img_data in self.annotations["imgs"].items():
            img_path = str(Path(self.images_folder) / img_data["file_name"])
            if not os.path.exists(img_path):
                LOGGER.info(
                    "Can't open filepath %s, it could not exist or be corrupted.",
                    img_path,
                )
                continue
            valid_ids.append(img_id)
        return sorted(valid_ids)

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales an image by a specified factor.

        This function resizes the input image using the specified resizing method.

        Args:
            img (np.ndarray): The input image to be resized.
            scale_factor (float): The factor by which to scale the image.
            resize_image_method (str, optional): The method to use for resizing the
                image (default is "bicubic").

        Returns:
            np.ndarray: The resized image (np.ndarray).
        """
        return RESIZE_METHODS.get(resize_image_method)(img, scale_factor), anns

    def reshape(self, img, anns, shape, resize_image_method="bicubic"):
        """
        Reshapes an image and its corresponding annotations to a specified size.

        This function resizes the input image and scales the bounding boxes,
        segmentation and all annotations according to the new dimensions. It offers the
        option to specify the method used for resizing the image.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (list[dict]): A list of annotations corresponding to the image,
                each containing at least a "bbox" key and optionally "segmentation".
            shape (tuple): The desired output shape of the image as (width, height).
            resize_image_method (str, optional): The method used for resizing the
                image, default is "bicubic". Other methods can be defined in
                RESIZE_METHODS.

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and the updated
            annotations (list[dict]).
        """
        return RESIZE_METHODS.get(resize_image_method)(img, shape), anns

    def show(self, image, anns):
        """
        Displays an image with its caption.

        This function takes an image represented as a NumPy array and a caption,
        visualizing the image with the caption.

        Args:
            image (np.ndarray): The image to display.
            caption (str): The caption to display with the image.

        Returns:
            None: This function does not return any value.
        """
        if not self._window:
            self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        for ann in anns:
            img_with_caption = image.copy()
            VISUALIZATION_REGISTRY.caption(img_with_caption, ann["caption"])
            cv2.imshow(self._window_name, img_with_caption)
            cv2.waitKey(1)

    def __getitem__(self, index):
        """
        Retrieves an image and its associated caption from the dataset based on the given index.

        This function accesses the dataset using the specified index to return the processed image
        and its corresponding caption.

        Args:
            index (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the processed image (np.ndarray) and the caption (str).
        """
        img_id = self.ids[index]
        anns = self.annotations["imgToAnns"].get(img_id, [])
        path = self.annotations["imgs"][img_id]["file_name"]
        img = cv2.imread(str(Path(self.images_folder) / path))
        resize_function, resize_parameter = (
            (self.scale, self.cfg.scale_factor)
            if self.cfg.image_shape is None
            else (self.reshape, self.cfg.image_shape)
        )
        img, anns = resize_function(
            img,
            anns,
            resize_parameter,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        if self.cfg.show:
            self.show(img, anns)
        return img, anns

    def save(self, index, image, anns):
        """
        Saves an image and its caption to the output folder.

        This function writes the image to the images output folder and saves the caption
        to a text file named according to the image ID.

        Args:
            index (int): The index of the image in the dataset.
            image (np.ndarray): The processed image to be saved.
            caption (str): The caption to be saved.

        Returns:
            None: This function does not return any value.
        """
        img_id = self.ids[index]
        img_ann = self.annotations.imgs[img_id]
        cv2.imwrite(str(Path(self.images_output_folder) / img_ann["file_name"]), image)
        img_ann["height"], img_ann["width"] = image.shape[:2]
        self.output_annotations["annotations"].extend(anns)
        self.output_annotations["images"].append(img_ann)

    def __len__(self):
        """
        Returns the number of images in the dataset.

        This function calculates and returns the total number of images available in
        the dataset by determining the length of the internal list of image identifiers.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.ids)

    def close(self):
        """
        Optionally, save any results or annotations if needed.

        For image captioning, this might not be required, but included for consistency.
        """
        save_json(
            self.output_annotations,
            self.labels_output_path,
        )
        if self._window:
            cv2.destroyWindow(self._window_name)


@COCO_TASKS.register(name="keypoints")
class COCODatasetKeypoints(ResizableDataset):
    """
    A dataset class for handling COCO-format keypoint annotations, supporting
    image scaling, annotation filtering, and image loading operations.

    Attributes:
        cfg (ConfigDict): Configuration dictionary containing paths and settings.
        images_folder (str): Path to the folder containing images.
        ids (list): List of valid image IDs.
        id2name (dict): Mapping from category ID to category name.
        name2id (dict): Mapping from category name to category ID.
        id2color (dict): Mapping from category ID to a unique color.
        id2cat (dict): Mapping from category ID to category details.
        images_output_folder (str): Path to the folder where processed images will be saved.
        labels_output_path (str): Path to the file where output annotations will be saved.
        output_annotations (dict): Dictionary containing the output annotations.

    Methods:
        __init__(images_path, annotations_path, cfg): Initializes the dataset with image
            paths, annotations, and configuration.
        _create_annotations(annotations_path): Creates and initializes keypoint annotations
            from a JSON file.
        _filter_valid_images(): Checks which images can be successfully loaded and returns
            a list of valid image IDs.
        _filter_valid_annotations(): Updates annotations to include only those related
            to valid images.
        scale(img, anns, scale_factor, resize_image_method="bicubic"): Scales the image
            and its keypoint annotations.
        save(index, image, anns): Saves the processed image and annotations to the
            specified output folder.
        show(image, anns): Displays the image with keypoint annotations.
        __getitem__(index): Retrieves and processes an image and its annotations by index.
        __len__(): Returns the number of valid images in the dataset.
        close(): Saves the output annotations to a file.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        # Validate images and filter out invalid indices
        self.ids = self._get_valid_image_ids()
        self.id2name = {k: v["name"] for k, v in self.annotations.cats.items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        # To save
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        # Initialize the updated COCO dictionary
        self.output_annotations = self.annotations.dataset
        self.output_annotations["annotations"] = []
        self.output_annotations["images"] = []
        if self.cfg.save:
            ensure_folder_exist(self.images_output_folder)
        # To show
        self._window_name = "Annotations"
        self._window = None

    def _create_annotations(self, annotations_path):
        """
        Creates and initializes keypoint annotations for a dataset from a JSON file.

        Args:
            annotations_path (str): Path to the JSON file containing annotations.
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["anns"], self.annotations["cats"], self.annotations["imgs"] = (
            {},
            {},
            {},
        )
        self.annotations["imgToAnns"], self.annotations["catToImgs"] = (
            ConfigDict(),
            ConfigDict(),
        )
        if "annotations" in self.annotations.dataset:
            for ann in self.annotations.dataset["annotations"]:
                if ann["image_id"] not in self.annotations.imgToAnns:
                    self.annotations.imgToAnns[ann["image_id"]] = []
                self.annotations.imgToAnns[ann["image_id"]].append(ann)
                self.annotations["anns"][ann["id"]] = ann
        if "images" in self.annotations.dataset:
            for img in self.annotations.dataset["images"]:
                self.annotations["imgs"][img["id"]] = img
        if "categories" in self.annotations.dataset:
            for cat in self.annotations.dataset["categories"]:
                self.annotations["cats"][cat["id"]] = cat
        # Populate category to images mapping
        for ann in self.annotations["anns"].values():
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if cat_id not in self.annotations["catToImgs"]:
                self.annotations["catToImgs"][cat_id] = []
            self.annotations["catToImgs"][cat_id].append(img_id)

    def _get_valid_image_ids(self):
        """
        Checks which images can be successfully loaded and returns a list of valid image IDs.

        Returns:
            list: A list of valid image IDs.
        """
        valid_ids = []
        for img_id, img_data in self.annotations["imgs"].items():
            img_path = str(Path(self.images_folder) / img_data["file_name"])
            if not os.path.exists(img_path):
                LOGGER.info(
                    "Can't open filepath %s, it could not exist or be corrupted.",
                    img_path,
                )
                continue
            valid_ids.append(img_id)
        return sorted(valid_ids)

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales the image and its keypoint annotations.

        Args:
            img (np.ndarray): The image to be scaled.
            anns (list): List of annotations associated with the image.
            scale_factor (float): Factor by which to scale the image and keypoints.
            resize_image_method (str): Method used to resize the image. Defaults to "bicubic".

        Returns:
            tuple: Scaled image and annotations.
        """
        img = RESIZE_METHODS.get(resize_image_method)(img, scale_factor)
        for ann in anns:
            if "bbox" in ann:
                ann["bbox"] = [c * scale_factor for c in ann["bbox"]]
            ann["area"] *= scale_factor**2
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):  # Segmentation polygon
                    for i, polygon in enumerate(ann["segmentation"]):
                        ann["segmentation"][i] = [c * scale_factor for c in polygon]
                else:
                    seg = ann["segmentation"]
                    h, w = seg["size"]
                    if isinstance(ann["segmentation"]["counts"], str):
                        mask = mask_utils.decode(ann["segmentation"])
                    else:
                        mask = rle_to_mask(seg["counts"], h, w, order="F")
                    resized_mask = cv2.resize(
                        mask,
                        (w * scale_factor, h * scale_factor),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    if isinstance(ann["segmentation"]["counts"], str):
                        ann["segmentation"]["counts"] = binary_mask_to_rle_coded(
                            resized_mask
                        )
                    else:
                        ann["segmentation"]["counts"] = mask_to_rle(
                            resized_mask, order="F"
                        )
                    ann["segmentation"]["size"] = [
                        h * scale_factor,
                        w * scale_factor,
                    ]
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                scaled_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i] * scale_factor
                    y = keypoints[i + 1] * scale_factor
                    v = keypoints[i + 2]  # visibility flag remains the same
                    scaled_keypoints.extend([x, y, v])
                ann["keypoints"] = scaled_keypoints
        return img, anns

    def reshape(self, img, anns, shape, resize_image_method="bicubic"):
        """
        Reshapes an image and its corresponding annotations to a specified size.

        This function resizes the input image and scales the bounding boxes,
        segmentation and all annotations according to the new dimensions. It offers the
        option to specify the method used for resizing the image.

        Args:
            img (np.ndarray): The input image to be resized.
            anns (list[dict]): A list of annotations corresponding to the image,
                each containing at least a "bbox" key and optionally "segmentation".
            shape (tuple): The desired output shape of the image as (width, height).
            resize_image_method (str, optional): The method used for resizing the
                image, default is "bicubic". Other methods can be defined in
                RESIZE_METHODS.

        Returns:
            tuple: A tuple containing the resized image (np.ndarray) and the updated
            annotations (list[dict]).
        """
        h0, w0 = img.shape[:2]
        wn, hn = shape
        xf = wn / w0
        yf = hn / h0
        img = RESIZE_METHODS.get(resize_image_method)(img, shape)
        for ann in anns:
            if "bbox" in ann:
                x1, y1, w, h = ann["bbox"]
                ann["bbox"] = [x1 * xf, y1 * yf, w * xf, h * yf]
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                scaled_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i] * xf
                    y = keypoints[i + 1] * yf
                    v = keypoints[i + 2]
                    scaled_keypoints.extend([x, y, v])
                ann["keypoints"] = scaled_keypoints
            ann["area"] *= xf * yf
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):
                    for i, polygon in enumerate(ann["segmentation"]):
                        ann["segmentation"][i] = [
                            c * xf if i % 2 == 0 else c * yf
                            for i, c in enumerate(polygon)
                        ]
                else:
                    seg = ann["segmentation"]
                    h0, w0 = seg["size"]
                    xf = wn / w0
                    yf = hn / h0
                    if isinstance(ann["segmentation"]["counts"], str):
                        mask = mask_utils.decode(ann["segmentation"])
                    else:
                        mask = rle_to_mask(seg["counts"], h0, w0, order="F")
                    resized_mask = cv2.resize(
                        mask, shape, interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)
                    if isinstance(ann["segmentation"]["counts"], str):
                        ann["segmentation"]["counts"] = binary_mask_to_rle_coded(
                            resized_mask
                        )
                    else:
                        ann["segmentation"]["counts"] = mask_to_rle(
                            resized_mask, order="F"
                        )
                    ann["segmentation"]["size"] = [hn, wn]
        return img, anns

    def save(self, index, image, anns):
        """
        Saves the processed image and annotations to the specified output folder.

        Args:
            index (int): Index of the image in the dataset.
            image (ndarray): Processed image to be saved.
            anns (list): List of processed annotations.
        """
        img_id = self.ids[index]
        img_ann = self.annotations.imgs[img_id]
        cv2.imwrite(str(Path(self.images_output_folder) / img_ann["file_name"]), image)
        img_ann["height"], img_ann["width"] = image.shape[:2]
        self.output_annotations["annotations"].extend(anns)
        self.output_annotations["images"].append(img_ann)

    def show(self, image, anns):
        """
        Displays the image with keypoint annotations.

        Args:
            image (np.ndarray): Image to be displayed.
            anns (list): List of annotations to be displayed on the image.
        """
        img_with_annotations = image.copy()
        colors = generate_n_unique_colors(len(anns))
        for i, ann in enumerate(anns):
            if "keypoints" in ann:
                category_info = self.annotations.cats.get(ann["category_id"], {})
                skeleton = np.array(category_info.get("skeleton", [])) - 1
                label = {
                    "keypoints": ann["keypoints"],
                    "skeleton": skeleton,
                    "segmentation": ann.get("segmentation", None),
                }
                img_with_annotations = VISUALIZATION_REGISTRY.keypoints(
                    img_with_annotations,
                    label,
                    color=colors[i],
                )
        if not self._window:
            self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self._window_name, img_with_annotations)
        cv2.waitKey(0)

    def __getitem__(self, index):
        """
        Retrieves and processes an image and its annotations by index.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            tuple: Processed image and annotations.
        """
        img_id = self.ids[index]
        anns = self.annotations["imgToAnns"].get(img_id, [])
        path = self.annotations["imgs"][img_id]["file_name"]
        img = cv2.imread(str(Path(self.images_folder) / path))
        resize_function, resize_parameter = (
            (self.scale, self.cfg.scale_factor)
            if self.cfg.image_shape is None
            else (self.reshape, self.cfg.image_shape)
        )
        img, anns = resize_function(
            img,
            anns,
            resize_parameter,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        if self.cfg.show:
            self.show(img, anns)
        return img, anns

    def __len__(self):
        """
        Returns the number of valid images in the dataset.

        Returns:
            int: Number of valid images.
        """
        return len(self.ids)

    def close(self):
        """
        Saves the output annotations to a file.
        """
        save_json(self.output_annotations, self.labels_output_path)
        if self._window:
            cv2.destroyWindow(self._window_name)
