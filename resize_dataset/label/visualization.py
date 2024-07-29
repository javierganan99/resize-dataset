import cv2
import numpy as np
from resize_dataset.utils import ConfigDict


def add_bounding_box(
    image, bbox, label=None, color=(255, 0, 0), text_color=(255, 255, 255)
):
    """
    Draws a bounding box on the given image with an optional label.

    This function adds a rectangular bounding box to an image, which can also include a label.
    The bounding box color and text color can be customized. The width of the bounding box is
    calculated based on the size of the image to ensure it is visually appropriate.

    Args:
        image (numpy.ndarray): The image on which to draw the bounding box.
        bbox (tuple): A tuple containing (x, y, width, height) of the bounding box.
        label (str, optional): The label text to be displayed. Defaults to None.
        color (tuple, optional): The color of the bounding box in BGR format.
            Default is (255, 0, 0).
        text_color (tuple, optional): The color of the label text in BGR format.
            Default is (255, 255, 255).

    Returns:
        numpy.ndarray: The image with the bounding box and label drawn on it.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale
    p1, p2 = (int(bbox[0]), int(bbox[1])), (
        int(bbox[0]) + int(bbox[2]),
        int(bbox[1]) + int(bbox[3]),
    )
    cv2.rectangle(
        image,
        p1,
        p2,
        color=color,
        thickness=lw,
    )
    if label is not None:
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            sf,
            text_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return image


def add_polygons(
    image,
    polygons,
    label=None,
    color=(255, 0, 0),
    alpha=0.25,
    text_color=(255, 255, 255),
):
    # TODO: CORRECT TRANSPARENCY AS IN add_mask!!!!!!!
    """
    Draws segmentation polygons on the image with optional labels and fills them with transparency.

    This function overlays the given polygons on the specified image and allows
    for adjustable transparency, color, and optional text labels for each polygon.
    The polygons are filled with a specified color and transparency level, and if
    a label is provided, it is drawn on the image in a specified text color.

    Args:
        image (np.ndarray): The image on which to draw the polygons.
        polygons (list): List of segmentation polygons, where each polygon is a list of coordinates.
        label (str, optional): Optional. The label to display with the polygon.
        color (tuple): The color for the polygon (B, G, R) (default is (255, 0, 0)).
        alpha (float): Transparency factor for the polygon fill (0.0 to 1.0) (default is 0.25).
        text_color (tuple): The color for the label text (B, G, R) (default is (255, 255, 255)).

    Returns:
        np.ndarray: The image with drawn polygons.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    overlay = image.copy()
    for polygon in polygons:
        polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [polygon], color=color)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=lw)
    # Apply the overlay with transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    if label is not None:
        cX, cY = np.mean(
            np.array(polygons[0]).reshape(-1, 2).astype(np.int32), axis=0
        ).astype(int)
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[
            0
        ]  # text width, height
        p1 = (cX, cY)
        p2 = cX + w, cY - h - 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (cX, cY - 2),
            0,
            sf,
            text_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return image


def add_mask(
    image, mask, label=None, color=(255, 0, 0), alpha=0.5, text_color=(255, 255, 255)
):
    """
    Adds a mask to the image with optional label.

    This function overlays a given mask on the provided image, allowing for customization
    of the mask's color, transparency, and optional labeling. The mask is applied using a
    weighted addition approach, and the label can be displayed at the center of the masked
    area.

    Args:
        image (np.ndarray): The image on which to draw the mask.
        mask (np.ndarray): Boolean mask array with shape HxW.
        label (str, optional): Optional. The label to display with the mask.
        color (tuple): The color for the mask (B, G, R).
        alpha (float): Transparency factor for the mask fill (0.0 to 1.0).
        text_color (tuple): The color for the label text (B, G, R).

    Returns:
        np.ndarray: The image with the mask drawn.
    """
    mask = mask.astype(bool)  # Ensure mask is boolean
    image[mask] = (
        image[mask].astype(np.float32) * (1 - alpha) + np.array(color) * alpha
    ).astype("uint8")
    if label is not None:
        # Find center of the mask
        mask_indices = np.where(mask)
        cY, cX = np.mean(mask_indices, axis=1).astype(int)
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[
            0
        ]  # text width, height
        p1 = (cX, cY)
        p2 = cX + w, cY - h - 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (cX, cY - 2),
            0,
            sf,
            text_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return image


VISUALIZATION_REGISTRY = ConfigDict(
    bbox=add_bounding_box, mask=add_mask, polygons=add_polygons
)
