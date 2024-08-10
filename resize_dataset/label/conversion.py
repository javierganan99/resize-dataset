import numpy as np
from pycocotools import _mask as coco_mask


def rle_to_mask(rle, height, width, order="F"):
    """
    Converts a Run-Length Encoded (RLE) mask to a 2D numpy array.

    This function takes an RLE representation of a binary mask, which specifies the
    locations and lengths of runs of pixels set to 1, and converts it into a 2D
    numpy array of the specified height and width.

    Args:
        rle (list): Run-length encoded mask in the format
            [start1, length1, start2, length2, ...], where 'start' indicates
            the number of pixels to skip from the last position of 1.
        height (int): Height of the resulting mask.
        width (int): Width of the resulting mask.
        order (str, optional): Multi-dimensional data layout order. Default is 'F'
            (Fortran-style column-major order).

    Returns:
        np.ndarray: The decoded mask represented as a 2D numpy array.
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    rle_array = np.array(rle, dtype=np.int32).squeeze()
    starts = rle_array[0::2]
    lengths = rle_array[1::2]
    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position : current_position + length] = 1
        current_position += length  # Move the current position forward
    return mask.reshape((height, width), order=order)


def mask_to_rle(mask, order="F"):
    """
    Converts a 2D numpy array mask to RLE (Run-Length Encoded) format.

    This function takes a binary mask (2D numpy array) and encodes it using
    Run-Length Encoding (RLE). The RLE format is useful for compression and
    efficient storage of sparse binary images.

    Args:
        mask (np.ndarray): The 2D mask array to be encoded.
        order (str, optional): The order in which to read the mask.
            'F' for Fortran order (column-major), 'C' for C order (row-major).
            Default is 'F'.

    Returns:
        list: The RLE encoded mask in the format [start1, length1, start2,
            length2, ...] where start is relative to the last position of 1.
    """
    pixels = mask.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[2::2] -= runs[:-2:2] + runs[1:-2:2]
    return runs.tolist()


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
    detection = {"size": [im_height, im_width], "counts": rle_coded_str}
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
    return encoded_mask.decode("utf-8")
