import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import zlib
from resize_dataset.utils import ConfigDict
import pycocotools.mask as mask_util


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


def add_keypoints(
    image,
    keypoints,
    label=None,
    keypoint_color=(0, 255, 0),
    skeleton_color=(255, 0, 0),
    keypoint_radius=5,
    skeleton_thickness=2,
):
    """
    Draw keypoints and optional skeleton on an image.

    Parameters:
    - image (ndarray): The image to which keypoints and skeleton will be drawn.
    - keypoints (list or ndarray): A list or array of keypoints, where each keypoint is a tuple of (x, y, visibility).
    - label (dict, optional): A dictionary containing the skeleton structure if available.
      It should have a key "skeleton" which is a list of tuples, where each tuple contains two indices (start_joint, end_joint).
    - keypoint_color (tuple, optional): Color of the keypoints in BGR format. Default is green (0, 255, 0).
    - skeleton_color (tuple, optional): Color of the skeleton lines in BGR format. Default is red (255, 0, 0).
    - keypoint_radius (int, optional): Radius of the keypoints to be drawn. Default is 5.
    - skeleton_thickness (int, optional): Thickness of the skeleton lines. Default is 2.

    Returns:
    - ndarray: The image with keypoints and skeleton drawn on it.
    """
    keypoints = np.array(keypoints).reshape(-1, 3)

    for x, y, v in keypoints:
        if v > 0:
            cv2.circle(image, (int(x), int(y)), keypoint_radius, keypoint_color, -1)

    if label and "skeleton" in label:
        skeleton = label["skeleton"]
        num_keypoints = keypoints.shape[0]
        for joint in skeleton:
            if joint[0] < num_keypoints and joint[1] < num_keypoints:
                x1, y1, v1 = keypoints[joint[0]]
                x2, y2, v2 = keypoints[joint[1]]
                if v1 > 0 and v2 > 0:
                    cv2.line(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        skeleton_color,
                        skeleton_thickness,
                    )
    return image


def add_dense_poses(image, anns):
    """
    Creates a figure with subplots showing only DensePose annotations.

    Args:
        image (np.ndarray): The input image to annotate.
        anns (list of dict): List of annotations containing DensePose data.

    Returns:
        fig (matplotlib.figure.Figure): The resulting figure with annotations.
    """

    fig, axs = plt.subplots(1, 3, figsize=[15, 5])

    extent = [0, image.shape[1], image.shape[0], 0]

    axs[0].imshow(image, extent=extent)
    axs[0].axis("off")
    axs[0].set_title("Patch Indices")

    axs[1].imshow(image, extent=extent)
    axs[1].axis("off")
    axs[1].set_title("U Coordinates")

    axs[2].imshow(image, extent=extent)
    axs[2].axis("off")
    axs[2].set_title("V Coordinates")

    for ann in anns:
        bbr = np.round(ann["bbox"]).astype(int)
        if "dp_masks" in ann:
            Point_x = np.array(ann["dp_x"]) / 255.0 * bbr[2]
            Point_y = np.array(ann["dp_y"]) / 255.0 * bbr[3]
            Point_I = np.array(ann["dp_I"])
            Point_U = np.array(ann["dp_U"])
            Point_V = np.array(ann["dp_V"])

            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            x2 = min(x2, image.shape[1])
            y2 = min(y2, image.shape[0])

            Point_x += x1
            Point_y += y1

            plt.subplot(1, 3, 1)
            plt.scatter(Point_x, Point_y, 22, Point_I)
            plt.subplot(1, 3, 2)
            plt.scatter(Point_x, Point_y, 22, Point_U)
            plt.subplot(1, 3, 3)
            plt.scatter(Point_x, Point_y, 22, Point_V)
            for ax in axs:
                ax.add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        bbr[2],
                        bbr[3],
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
    for ax in axs:
        ax.set_aspect("equal")

    return fig


VISUALIZATION_REGISTRY = ConfigDict(
    bbox=add_bounding_box,
    mask=add_mask,
    polygons=add_polygons,
    keypoints=add_keypoints,
    densepose=add_dense_poses,
)
