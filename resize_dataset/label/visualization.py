import cv2
import numpy as np
from resize_dataset.utils import ConfigDict
from .conversion import rle_coded_to_binary_mask, rle_to_mask

VISUALIZATION_REGISTRY = ConfigDict()


@VISUALIZATION_REGISTRY.register(name="bbox")
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


@VISUALIZATION_REGISTRY.register(name="polygons")
def add_polygons(
    image,
    polygons,
    label=None,
    color=(255, 0, 0),
    alpha=0.25,
    text_color=(255, 255, 255),
    line_width=None,
):
    """
    Draws segmentation polygons on the image with optional labels and fills them with transparency.

    This function overlays the given polygons on the specified image, allowing for adjustable
    transparency, color, and optional text labels for each polygon. The polygons are filled
    with a specified color and transparency level, and if a label is provided, it is drawn on
    the image in a specified text color.

    Args:
        image (np.ndarray): The image on which to draw the polygons.
        polygons (list): List of segmentation polygons, where each polygon is a list of coordinates.
        label (str, optional): Optional. The label to display with the polygon.
        color (tuple): The color for the polygon (B, G, R) (default is (255, 0, 0)).
        alpha (float): Transparency factor for the polygon fill (0.0 to 1.0) (default is 0.25).
        text_color (tuple): The color for the label text (B, G, R) (default is (255, 255, 255)).
        line_width (int, optional): Optional. The width of the polygon outline
            (default is calculated based on image size).

    Returns:
        np.ndarray: The image with drawn polygons.
    """
    lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
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


@VISUALIZATION_REGISTRY.register(name="mask")
def add_mask(
    image, mask, label=None, color=(255, 0, 0), alpha=0.5, text_color=(255, 255, 255)
):
    """
    Adds a mask to the image with an optional label.

    This function overlays a specified mask on the provided image, allowing for customization
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


@VISUALIZATION_REGISTRY.register(name="keypoints")
def add_keypoints(
    image,
    annotation,
    color=(255, 0, 0),
):
    """
    Adds keypoints and segmentation to the provided image based on the given annotation.

    This function processes the annotation to retrieve keypoint positions and
    segmentation masks. It draws keypoints as filled circles and connects them
    with lines if a skeleton is defined. If segmentation is present, it can
    draw polygons or a binary mask onto the image.

    Args:
        image (np.nd): The image on which keypoints and segmentation will be drawn.
        annotation (dict): A dictionary containing keypoints, skeleton, and segmentation data.
        color (tuple, optional): A tuple representing the color in RGB format
            (default is (255, 0, 0)).

    Returns:
        np.nd: The modified image with keypoints and segmentation drawn on it.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    r = int(lw * 1.5)
    keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
    skeleton = annotation.get("skeleton", None)
    seg = annotation.get("segmentation", None)
    if seg is not None:
        if isinstance(seg, list):
            add_polygons(image, seg, color=color, line_width=int(lw / 2))
        else:
            h, w = seg["size"]
            if isinstance(seg["counts"], str):
                mask = rle_coded_to_binary_mask(seg["counts"], h, w)
            else:
                mask = rle_to_mask(seg["counts"], h, w, order="F")
            add_mask(image, mask, color=color)
    for x, y, v in keypoints:
        if v > 0:
            cv2.circle(image, (int(x), int(y)), r, color, -1)
    if skeleton is not None:
        for joint in skeleton:
            x1, y1, v1 = keypoints[joint[0]]
            x2, y2, v2 = keypoints[joint[1]]
            if v1 > 0 and v2 > 0:
                cv2.line(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    lw,
                )
    return image


@VISUALIZATION_REGISTRY.register(name="densepose")
def add_dense_poses(image, anns):
    """
    Creates an image showing DensePose annotations stacked horizontally.

    This function processes an input image and overlays DensePose annotations,
    including patch indices, U coordinates, and V coordinates, on top of it.
    The annotations are visualized as circles representing different features
    based on the provided annotations data.

    Args:
        image (np.ndarray): The input image to annotate.
        anns (list of dict): List of annotations containing DensePose data.

    Returns:
        np.ndarray: The resulting image with annotations stacked horizontally.
    """
    # Create copies of the original image for each annotation type
    h, w = image.shape[:2]
    ps = int(min(h, w) * 0.01)
    patch_indices_image = image.copy()
    u_coordinates_image = image.copy()
    v_coordinates_image = image.copy()
    for ann in anns:
        bbr = np.round(ann["bbox"]).astype(int)
        if "dp_masks" in ann:
            point_x = np.array(ann["dp_x"]) / 255.0 * bbr[2]
            point_y = np.array(ann["dp_y"]) / 255.0 * bbr[3]
            point_i = apply_colormap(np.array(ann.get("dp_I", [])) / 24.0)
            point_u = apply_colormap(np.array(ann.get("dp_U", [])))
            point_v = apply_colormap(np.array(ann.get("dp_V", [])))
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            x2 = min(x2, image.shape[1])
            y2 = min(y2, image.shape[0])
            point_x += x1
            point_y += y1
            # Convert to integer pixel positions
            point_x = point_x.astype(int)
            point_y = point_y.astype(int)
            for px, py, i, u, v in zip(point_x, point_y, point_i, point_u, point_v):
                cv2.circle(
                    patch_indices_image, (px, py), ps, tuple(int(i) for i in i), -1
                )
                cv2.circle(
                    u_coordinates_image, (px, py), ps, tuple(int(i) for i in u), -1
                )
                cv2.circle(
                    v_coordinates_image, (px, py), ps, tuple(int(i) for i in v), -1
                )
    add_caption(patch_indices_image, "Patch Indices")
    add_caption(u_coordinates_image, "U Coordinates")
    add_caption(v_coordinates_image, "V Coordinates")
    return np.hstack((patch_indices_image, u_coordinates_image, v_coordinates_image))


def draw_rounded_rectangle(image, top_left, bottom_right, color, radius, thickness=-1):
    """
    Draws a rounded rectangle on the given image.

    This function adds a rectangle with rounded corners to an image using specified
    color, corner radius, and thickness. It utilizes OpenCV functions to render both the
    corners and the edges of the rectangle.

    Args:
        image (ndarray): The image on which to draw the rectangle.
        top_left (tuple): The (x, y) coordinates of the top-left corner of the rectangle.
        bottom_right (tuple): The (x, y) coordinates of the bottom-right corner of the rectangle.
        color (tuple): The RGB color for the rectangle.
        radius (int): The radius of the corners of the rectangle.
        thickness (int, optional): The thickness of the rectangle edges. Default is -1 for filled.

    Returns:
        None: This function modifies the image in place and does not return a value.
    """
    corners = [
        (top_left[0] + radius, top_left[1] + radius),
        (top_left[0] + radius, bottom_right[1] - radius),
        (bottom_right[0] - radius, top_left[1] + radius),
        (bottom_right[0] - radius, bottom_right[1] - radius),
    ]
    for corner in corners:
        cv2.circle(image, corner, radius, color, thickness)
    cv2.rectangle(
        image,
        (top_left[0] + radius, top_left[1]),
        (bottom_right[0] - radius, bottom_right[1]),
        color,
        thickness,
    )
    cv2.rectangle(
        image,
        (top_left[0], top_left[1] + radius),
        (bottom_right[0], bottom_right[1] - radius),
        color,
        thickness,
    )


def apply_colormap(values):
    """
    Maps a normalized value to a color using a colormap.

    This function takes an array of normalized values (between 0 and 1) and applies a color
    mapping to produce corresponding BGR color values using OpenCV's applyColorMap function.

    Args:
        values (np.ndarray): A (n, 3) array of normalized values between 0 and 1 to map to colors.

    Returns:
        np.ndarray: A (n, 3) array representing the BGR color corresponding to the input values.
    """
    if len(values) == 0:
        return values
    return cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[
        :, 0, :
    ]


@VISUALIZATION_REGISTRY.register(name="caption")
def add_caption(
    image,
    text,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    box_color=(0, 10, 0),
    text_color=(255, 255, 255),
    opacity=0.7,
):
    """
    Adds a caption overlay to an image with customizable text, font, and colors.

    This function computes the optimal placement for the caption, ensuring it
    fits within the given image dimensions. It also blends a background box
    behind the text to enhance visibility, applying a specified opacity to
    the blend.

    Args:
        image (numpy.ndarray): The image to which the caption will be added.
        text (str): The text of the caption to overlay on the image.
        font (int, optional): The font type for the text (default is cv2.FONT_HERSHEY_SIMPLEX).
        box_color (tuple, optional): The color of the box behind the text in BGR format
            (default is (0, 10, 0)).
        text_color (tuple, optional): The color of the text in BGR format
            (default is (255, 255, 255)).
        opacity (float, optional): The opacity for the text overlay (default is 0.7).

    Returns:
        None: This function modifies the input image in place and does not return a value.
    """
    h, w = image.shape[:2]
    # Set caption size based on image dimensions
    scale = w * 0.003
    thickness = max(1, int(w * 0.002))
    margin = int(h * 0.02)
    radius = int(h * 0.015)
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (w - text_size[0]) // 2  # Center text horizontally
    text_y = h - margin  # Position text near the bottom
    while text_size[0] > w - 2 * margin:  # Ensure text fits within image width
        scale -= 0.1
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
    box_top_left = (text_x - margin, text_y - text_size[1] - margin)
    box_bottom_right = (text_x + text_size[0] + margin, text_y + margin)
    # Create an overlay image
    overlay = image.copy()
    # Draw the rounded rectangle on the overlay
    draw_rounded_rectangle(
        overlay, box_top_left, box_bottom_right, box_color, radius=radius
    )
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
    # Add the text on top of the blended image
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
