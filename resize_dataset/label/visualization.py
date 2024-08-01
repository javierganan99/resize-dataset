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


def add_dense_poses(image, ann, point_color=(0, 255, 0), point_radius=2):
    """
    Draw dense pose points on an image.

    Parameters:
    - image (ndarray): The image to which dense pose points will be drawn.
    - ann (dict): A dictionary containing dense pose annotations. It should have keys "densepose" with values being
      dictionaries containing "u" and "v" which are lists or arrays of u and v coordinates of the dense pose points.
    - point_color (tuple, optional): Color of the dense pose points in BGR format. Default is green (0, 255, 0).
    - point_radius (int, optional): Radius of the dense pose points to be drawn. Default is 2.

    Returns:
    - ndarray: The image with dense pose points drawn on it.
    """
    u_coords = np.array(ann["densepose"]["u"]).astype(int)
    v_coords = np.array(ann["densepose"]["v"]).astype(int)
    for u, v in zip(u_coords, v_coords):
        cv2.circle(image, (u, v), point_radius, point_color, -1)
    return image


VISUALIZATION_REGISTRY = ConfigDict(
    bbox=add_bounding_box,
    mask=add_mask,
    polygons=add_polygons,
    keypoints=add_keypoints,
    dense_pose=add_dense_poses,
)
