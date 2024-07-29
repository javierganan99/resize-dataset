import cv2
from resize_dataset.utils import ConfigDict


def resize_image_bicubic(image, scale):
    """
    Resizes an image using bicubic interpolation.

    This function takes an image and a scaling factor, then resizes the image to
    the specified scale using bicubic interpolation, which produces smoother results
    compared to other interpolation methods.

    Args:
        image (numpy.ndarray): The input image to be resized,
            expected in height x width x channels format.
        scale (float): The scaling factor to resize the image.

    Returns:
        numpy.ndarray: The resized image with the new dimensions.
    """
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )


RESIZE_METHODS = ConfigDict(bicubic=resize_image_bicubic)
