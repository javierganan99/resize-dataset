from typing import overload, Union, Tuple
import cv2
import numpy as np
from resize_dataset.utils import ConfigDict

RESIZE_METHODS = ConfigDict()


def infer_type_and_load_arguments(
    *args, **kwargs
) -> Tuple[str, np.ndarray, Union[int, float, Tuple[int, int]]]:
    """
    Infers the type of operation (resize or scale) and loads arguments accordingly.

    Args:
        *args: Positional arguments which can be an image, and optionally shape or scale.
        **kwargs: Keyword arguments which can include 'image', 'shape', and 'scale'.

    Returns:
        tuple: A tuple containing:
            - function_type (str): The type of operation ('resize' or 'scale').
            - image (np.ndarray): The image to be processed.
            - other (Union[int, float, tuple]): The scale or shape for resizing.

    Raises:
        ValueError: If required parameters are missing or if both shape and scale are provided.
        TypeError: If input types are incorrect.
    """
    if len(args) == 1:
        image = args[0]
        if not isinstance(image, np.ndarray):
            raise TypeError("The image should be a numpy ndarray.")
    elif len(args) == 2:
        image, other = args
        if not isinstance(image, np.ndarray):
            raise TypeError("The image should be a numpy ndarray.")
        if isinstance(other, (list, tuple, np.ndarray)):
            function_type = "resize"
        elif isinstance(other, (int, float)):
            function_type = "scale"
        else:
            raise TypeError(
                "The second argument must be a list, tuple, ndarray, int, or float."
            )
        return function_type, image, other
    elif len(args) == 0:
        image = kwargs.get("image", None)
        if image is None:
            raise ValueError("Image parameter missing!")
        if not isinstance(image, np.ndarray):
            raise TypeError("The image should be a numpy ndarray.")
    else:
        raise ValueError("Invalid number of arguments provided.")
    shape = kwargs.get("shape", None)
    scale = kwargs.get("scale", None)
    if shape is not None:
        function_type = "resize"
        other = shape
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")
    elif scale is not None:
        function_type = "scale"
        other = scale
        if not isinstance(scale, (int, float)):
            raise TypeError("Scale must be an int or float.")
    else:
        raise ValueError("Scale or shape parameter missing!")
    return function_type, image, other


@overload
def resize_image_bicubic(image: np.ndarray, scale: Union[float, int]) -> np.ndarray: ...
@overload
def resize_image_bicubic(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray: ...


def _scale_image_bicubic(image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
    """
    Scales an image using bicubic interpolation.

    Args:
        image (np.ndarray): The image to be scaled.
        scale (Union[float, int]): The scale factor.

    Returns:
        np.ndarray: The scaled image.
    """
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )


def _reshape_image_bicubic(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Reshapes an image to the given dimensions using bicubic interpolation.

    Args:
        image (np.ndarray): The image to be reshaped.
        shape (tuple): The target dimensions (width, height).

    Returns:
        np.ndarray: The reshaped image.
    """
    return cv2.resize(image, shape, interpolation=cv2.INTER_CUBIC)


@RESIZE_METHODS.register(name="bicubic")
def resize_image_bicubic(*args, **kwargs) -> np.ndarray:
    """
    Resizes or scales an image using bicubic interpolation.

    Args:
        *args: Positional arguments which can be an image, and optionally shape or scale.
        **kwargs: Keyword arguments which can include 'image', 'shape', and 'scale'.

    Returns:
        np.ndarray: The resized or scaled image.
    """
    function_type, image, other = infer_type_and_load_arguments(*args, **kwargs)
    if function_type == "scale":
        return _scale_image_bicubic(image, other)
    else:
        return _reshape_image_bicubic(image, other)
