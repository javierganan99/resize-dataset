import base64
import zlib
from pathlib import Path
import cv2
import torch
import numpy as np
import os
import pycocotools.mask as mask_utils
from pycocotools import _mask as coco_mask
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
from .utils import generate_n_unique_colors
from .base import ResizableDataset

COCO_TASKS = ConfigDict()


def rle_coded_to_binary_mask(rle_coded_str, im_height, im_width):
    """
    Converts a run-length encoded (RLE) string to a binary mask.

    This function decodes the input run-length encoded string, decompresses it,
    and constructs a binary mask based on the specified image dimensions. The
    resulting mask can be used for image segmentation tasks, particularly in
    computer vision applications.

    Args:
        rle_coded_str (str): The run-length encoded string representing the mask.
        im_height (int): The height of the image.
        im_width (int): The width of the image.

    Returns:
        numpy.ndarray: A binary mask of shape (im_height, im_width), where True
        values indicate the presence of the detected object(s) and False values
        indicate the absence.
    """
    uncoded_str = base64.b64decode(rle_coded_str)
    uncompressed_str = zlib.decompress(uncoded_str, wbits=zlib.MAX_WBITS)
    detection = {"size": [im_height, im_width], "counts": uncompressed_str}
    detlist = []
    detlist.append(detection)
    mask = coco_mask.decode(detlist)
    return mask[..., 0].astype("bool")


def binary_mask_to_rle_coded(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    This function takes a binary mask as input and converts it to a
    Run-Length Encoding format that is suitable for storage or transmission.
    The mask is first reshaped to ensure it has the correct dimensions,
    then it is encoded using the COCO mask encoding method. The encoded
    data is then compressed using zlib and base64 encoded for easier handling.

    Args:
        mask (numpy.ndarray): A binary mask array with shape (height, width)
            where the mask values are either 0 or 1.

    Returns:
        str: A base64 encoded string representing the compressed RLE format
            of the input mask.
    """
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    encoded_mask = coco_mask.encode(np.asfortranarray(mask))[0]["counts"]
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    return base64.b64encode(binary_str).decode()


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
        self.ids = list(sorted(self.annotations.imgs.keys()))
        self.id2name = {k: v["name"] for k, v in self.annotations.cats.items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        # To save
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        self.output_annotations = self.annotations.dataset

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
        image_name = self.annotations.imgs[img_id]["file_name"]
        cv2.imwrite(str(Path(self.images_output_folder) / image_name), image)
        self.output_annotations["annotations"].append(anns)

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
        cv2.namedWindow("Annotations", cv2.WINDOW_NORMAL)
        cv2.imshow("Annotations", img_with_annotations)
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
        img, anns = self.scale(
            img,
            anns,
            scale_factor=self.cfg.scale_factor,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        return img, [anns]

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
        self.ids = list(sorted(self.annotations.imgs.keys()))
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
        image_name = self.annotations.imgs[img_id]["file_name"]
        cv2.imwrite(str(Path(self.images_output_folder) / image_name), image)
        cv2.imwrite(
            str(Path(self.labels_output_folder) / annotation["file_name"]),
            scaled_mask,
        )
        self.output_annotations["annotations"].append(annotation)

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
        seg_img = scaled_mask[0].cpu().numpy().astype(np.uint32)
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
        cv2.namedWindow("Annotations", cv2.WINDOW_NORMAL)
        cv2.imshow("Annotations", img_with_annotations)
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
        img, anns = self.scale(
            img,
            anns,
            scale_factor=self.cfg.scale_factor,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        return img, [anns]

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


@COCO_TASKS.register(name="densepose")
class COCODatasetDensePose(ResizableDataset):
    """
    Dataset class for loading and processing images and DensePose annotations in COCO format.

    This class provides functionality to initialize a dataset from a specified images path
    and annotations path, retrieve images and DensePose annotations, scale images and annotations
    based on a given factor, and visualize the DensePose annotations on images.

    Args:
        images_path (str): Path to the folder containing images.
        annotations_path (str): Path to the COCO annotations JSON file.
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.

    Attributes:
        cfg (ConfigDict): Configuration object containing scaling and resizing parameters.
        images_folder (str): Path to the folder containing images.
        annotations (ConfigDict): DensePose-specific annotations.
        ids (list): Sorted list of image IDs in the annotations.
        id2name (dict): Mapping from category ID to category name.
        name2id (dict): Mapping from category name to category ID.
        id2color (dict): Mapping from category ID to a unique color.

    Methods:
        scale(img, anns, scale_factor, resize_image_method="bicubic"): Scales the DensePose annotations
            by the given scale factor.
        show(image, anns): Displays the image with its corresponding DensePose annotations.
        __getitem__(index): Retrieves the image and DensePose annotations for the given index.
        __len__(): Returns the total number of images in the dataset.

    Private Methods:
        _create_annotations(annotations_path): Initializes the DensePose annotations from the COCO format
            JSON file and sets up internal mappings.
        save(index, image, anns): Saves an image and its DensePose annotations to the output folder
            and dictionary.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        self.ids = list(sorted(self.annotations["imgs"].keys()))
        self.id2name = {k: v["name"] for k, v in self.annotations["cats"].items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        # To save
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        self.output_annotations = {"images": [], "annotations": []}

    def _create_annotations(self, annotations_path):
        """
        Creates and organizes DensePose annotations from a specified JSON file.

        This function loads DensePose annotations from a given path and structures them into
        a dictionary-like object. It organizes the annotations by mapping image IDs to their
        corresponding DensePose data.
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["anns"] = {}
        self.annotations["cats"] = {}
        self.annotations["imgs"] = {}
        self.annotations["imgToAnns"] = ConfigDict()
        self.annotations["catToImgs"] = ConfigDict()

        if "annotations" in self.annotations["dataset"]:
            for ann in self.annotations["dataset"]["annotations"]:
                if "densepose" in ann:
                    if ann["image_id"] not in self.annotations["imgToAnns"]:
                        self.annotations["imgToAnns"][ann["image_id"]] = []
                    self.annotations["imgToAnns"][ann["image_id"]].append(ann)
                    self.annotations["anns"][ann["id"]] = ann

        if "images" in self.annotations["dataset"]:
            for img in self.annotations["dataset"]["images"]:
                self.annotations["imgs"][img["id"]] = img

        if "categories" in self.annotations["dataset"]:
            for cat in self.annotations["dataset"]["categories"]:
                self.annotations["cats"][cat["id"]] = cat

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales DensePose annotations by a specified factor.

        This function scales the DensePose coordinates accordingly. The image scaling is
        handled by the base class, but DensePose-specific scaling is done here.
        """
        img = RESIZE_METHODS.get(resize_image_method)(img, scale_factor)
        for ann in anns:
            if "densepose" in ann:
                # Scale DensePose coordinates (u, v)
                ann["densepose"]["u"] = [
                    c * scale_factor for c in ann["densepose"]["u"]
                ]
                ann["densepose"]["v"] = [
                    c * scale_factor for c in ann["densepose"]["v"]
                ]
        return img, anns

    def show(self, image, anns):
        """
        Displays an image with DensePose annotations such as DensePose coordinates.

        This function takes an image represented as a NumPy array and a list of annotation
        dictionaries, visualizing the DensePose annotations on the image.
        """
        img_with_annotations = image.copy()
        # TODO: Include in Visualization registry
        for ann in anns:
            if "densepose" in ann:
                u_coords = np.array(ann["densepose"]["u"]).astype(int)
                v_coords = np.array(ann["densepose"]["v"]).astype(int)
                for u, v in zip(u_coords, v_coords):
                    cv2.circle(
                        img_with_annotations, (u, v), 2, (0, 255, 0), -1
                    )  # Green dots

        cv2.namedWindow("Annotations", cv2.WINDOW_NORMAL)
        cv2.imshow("Annotations", img_with_annotations)
        cv2.waitKey(0)

    def __getitem__(self, index):
        """
        Retrieves an image and its associated DensePose annotations from the dataset based on the given index.

        This function accesses the dataset using the specified index to return the processed image
        and its corresponding DensePose annotations.
        """
        img_id = self.ids[index]
        ann_ids = (
            [ann["id"] for ann in self.annotations["imgToAnns"].get(img_id, [])]
            if img_id in self.annotations["imgToAnns"]
            else []
        )
        anns = [self.annotations["anns"].get(idx, {}) for idx in ann_ids]
        path = self.annotations["imgs"][img_id]["file_name"]
        img = cv2.imread(str(Path(self.images_folder) / path))
        img, anns = self.scale(
            img,
            anns,
            scale_factor=self.cfg.scale_factor,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        return img, [anns]

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.ids)

    def save(self, index, image, anns):
        """
        Saves an image and its DensePose annotations to the output folder and dictionary.

        This function writes the image to the images output folder and appends
        the DensePose annotations to the output_annotations dictionary.
        """
        img_id = self.ids[index]
        image_name = self.annotations["imgs"][img_id]["file_name"]
        cv2.imwrite(str(Path(self.images_output_folder) / image_name), image)
        for ann in anns:
            if "densepose" in ann:
                self.output_annotations["annotations"].append(ann)
        self.output_annotations["images"].append(self.annotations["imgs"][img_id])

    def close(self):
        """
        Saves all DensePose annotations to a JSON file specified in the labels_output_path.
        """
        save_json(self.output_annotations, self.labels_output_path)


@COCO_TASKS.register(name="captioning")
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
        scale(img, scale_factor, resize_image_method="bicubic"): Scales the image by the given scale factor.
        show(image, caption): Displays the image with its corresponding caption.
        __getitem__(index): Retrieves the image and caption for the given index.
        __len__(): Returns the total number of images in the dataset.

    Private Methods:
        _create_annotations(annotations_path): Initializes the annotations from the COCO format JSON file.
    """

    def __init__(self, images_path, annotations_path, cfg):
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        self.ids = list(sorted(self.annotations.imgs.keys()))
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        self.output_annotations = self.annotations.dataset
        self.captions = {img_id: [] for img_id in self.ids}

    def _create_annotations(self, annotations_path):
        """
        Creates and organizes annotations from a specified JSON file.

        This function loads annotations from a given path and structures them into
        a dictionary-like object. It organizes the annotations by mapping image IDs
        to their corresponding captions.

        Args:
            annotations_path (str): The file path to the JSON annotations file.

        Returns:
            None: This function does not return a value. It modifies the
            annotations attribute of the class instance in place.
        """
        self.annotations = ConfigDict()
        self.annotations["dataset"] = load_json(annotations_path)
        self.annotations["imgs"] = {}
        self.captions = {}

        if "images" in self.annotations.dataset:
            for img in self.annotations.dataset["images"]:
                img_id = img["id"]
                self.annotations["imgs"][img_id] = img
                self.captions[img_id] = []

        if "annotations" in self.annotations.dataset:
            for ann in self.annotations.dataset["annotations"]:
                img_id = ann["image_id"]
                if img_id in self.captions:
                    self.captions[img_id].append(ann["caption"])

    def scale(self, img, scale_factor, resize_image_method="bicubic"):
        """
        Scales an image by a specified factor.

        This function resizes the input image using the specified resizing method.

        Args:
            img (np.ndarray): The input image to be resized.
            scale_factor (float): The factor by which to scale the image.
            resize_image_method (str, optional): The method to use for resizing the image (default is "bicubic").

        Returns:
            np.ndarray: The resized image (np.ndarray).
        """
        return RESIZE_METHODS.get(resize_image_method)(img, scale_factor)

    def show(self, image, caption):
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
        img_with_caption = image.copy()
        # You can use a library like PIL to draw the caption on the image if needed
        # For simplicity, we'll just display the image and caption as text
        cv2.namedWindow("Caption", cv2.WINDOW_NORMAL)
        cv2.imshow("Caption", img_with_caption)
        print(f"Caption: {caption}")
        cv2.waitKey(0)

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
        path = self.annotations.imgs[img_id]["file_name"]
        img = cv2.imread(str(Path(self.images_folder) / path))
        img = self.scale(
            img,
            scale_factor=self.cfg.scale_factor,
            resize_image_method=self.cfg.resize_image_method,
        )
        caption = self.captions[img_id][
            0
        ]  # Assuming there's one caption per image; modify as needed
        return img, caption

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
        image_name = self.annotations.imgs[img_id]["file_name"]
        cv2.imwrite(str(Path(self.images_output_folder) / image_name), image)
        self.output_annotations["annotations"].append(anns)

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


@COCO_TASKS.register(name="keypoint")
class COCODatasetKeypoint(ResizableDataset):
    """
    A dataset class for handling COCO-format keypoint annotations, supporting image scaling, annotation filtering,
    and image loading operations.

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
        __init__(images_path, annotations_path, cfg): Initializes the dataset with image paths, annotations, and configuration.
        _create_annotations(annotations_path): Creates and initializes keypoint annotations from a JSON file.
        _filter_valid_images(): Checks which images can be successfully loaded and returns a list of valid image IDs.
        _filter_valid_annotations(): Updates annotations to include only those related to valid images.
        scale(img, anns, scale_factor, resize_image_method="bicubic"): Scales the image and its keypoint annotations.
        save(index, image, anns): Saves the processed image and annotations to the specified output folder.
        show(image, anns): Displays the image with keypoint annotations.
        __getitem__(index): Retrieves and processes an image and its annotations by index.
        __len__(): Returns the number of valid images in the dataset.
        close(): Saves the output annotations to a file.
    """

    def __init__(self, images_path, annotations_path, cfg):
        """
        Initializes the COCODatasetKeypoint with image paths, annotations, and configuration.

        Args:
            images_path (str): Path to the folder containing images.
            annotations_path (str): Path to the JSON file containing annotations.
            cfg (ConfigDict): Configuration dictionary containing paths and settings.
        """
        self.cfg = cfg
        self.images_folder = images_path
        self._create_annotations(annotations_path)
        # Validate images and filter out invalid indices
        self.ids = self._filter_valid_images()
        self._filter_valid_annotations()  # New method to filter annotations
        self.id2name = {k: v["name"] for k, v in self.annotations.cats.items()}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.id2color = generate_n_unique_colors(self.id2name.keys())
        self.id2cat = self.annotations.cats

        # To save
        self.images_output_folder = self.cfg.images_output_path
        self.labels_output_path = self.cfg.labels_output_path
        # Initialize the updated COCO dictionary
        self.output_annotations = {
            "images": [],
            "annotations": [],
            "categories": list(self.annotations["cats"].values()),
        }
        if self.cfg.save:
            ensure_folder_exist(Path(self.images_output_folder))

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

    def _filter_valid_images(self):
        """
        Checks which images can be successfully loaded and returns a list of valid image IDs.

        Returns:
            list: A list of valid image IDs.
        """
        valid_ids = []
        for img_id in self.annotations.imgs.keys():
            path = self.annotations.imgs[img_id]["file_name"]
            img_path = str(Path(self.images_folder) / path)
            if not os.path.exists(img_path):
                LOGGER.info(
                    f"Can't open filepath {img_path}, it could not exist or be corrupted."
                )
                continue
            img = cv2.imread(img_path)
            if img is not None:
                valid_ids.append(img_id)
            else:
                LOGGER.info(
                    f"Can't open/read file: check file path/integrity {img_path}"
                )
        return valid_ids

    def _filter_valid_annotations(self):
        """
        Updates annotations to include only those related to valid images.
        """
        valid_img_ids = set(self.ids)

        self.annotations["imgs"] = {
            img_id: img
            for img_id, img in self.annotations["imgs"].items()
            if img_id in valid_img_ids
        }

        self.annotations["anns"] = {
            ann_id: ann
            for ann_id, ann in self.annotations["anns"].items()
            if ann["image_id"] in valid_img_ids
        }

        self.annotations["imgToAnns"] = {
            img_id: anns
            for img_id, anns in self.annotations["imgToAnns"].items()
            if img_id in valid_img_ids
        }

    def scale(self, img, anns, scale_factor, resize_image_method="bicubic"):
        """
        Scales the image and its keypoint annotations.

        Args:
            img (ndarray): The image to be scaled.
            anns (list): List of annotations associated with the image.
            scale_factor (float): Factor by which to scale the image and keypoints.
            resize_image_method (str): Method used to resize the image. Defaults to "bicubic".

        Returns:
            tuple: Scaled image and annotations.
        """
        img = RESIZE_METHODS.get(resize_image_method)(img, scale_factor)
        for ann in anns:
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

    def save(self, index, image, anns):
        """
        Saves the processed image and annotations to the specified output folder.

        Args:
            index (int): Index of the image in the dataset.
            image (ndarray): Processed image to be saved.
            anns (list): List of processed annotations.
        """
        img_id = self.ids[index]
        image_name = self.annotations.imgs[img_id]["file_name"]
        cv2.imwrite(str(Path(self.images_output_folder) / image_name), image)
        # Save the annotations in the new COCO format
        for ann in anns:
            if ann["id"] not in {
                a["id"] for a in self.output_annotations["annotations"]
            }:
                self.output_annotations["annotations"].append(ann)

        # Include the image info in the output dictionary
        if img_id not in {img["id"] for img in self.output_annotations["images"]}:
            self.output_annotations["images"].append(self.annotations.imgs[img_id])

    def show(self, image, anns):
        """
        Displays the image with keypoint annotations.

        Args:
            image (ndarray): Image to be displayed.
            anns (list): List of annotations to be displayed on the image.
        """
        img_with_annotations = image.copy()
        for ann in anns:
            if "keypoints" in ann:
                class_id = (
                    ann["category_id"].item()
                    if isinstance(ann["category_id"], torch.Tensor)
                    else ann["category_id"]
                )
                category_info = self.id2cat.get(class_id, {})
                label = {
                    "skeleton": category_info.get("skeleton", []),
                }
                print(f"category_info {category_info}")
                img_with_annotations = VISUALIZATION_REGISTRY.keypoints(
                    img_with_annotations, ann["keypoints"], label=label
                )
        cv2.namedWindow("Annotations", cv2.WINDOW_NORMAL)
        cv2.imshow("Annotations", img_with_annotations)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __getitem__(self, index):
        """
        Retrieves and processes an image and its annotations by index.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            tuple: Processed image and annotations.
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

        if img is None:
            raise RuntimeError(f"Failed to load image at path {path}")

        img, anns = self.scale(
            img,
            anns,
            scale_factor=self.cfg.scale_factor,
            resize_image_method=self.cfg.resize_image_method,
        )
        if self.cfg.save:
            self.save(index, img, anns)
        return img, [anns]

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
