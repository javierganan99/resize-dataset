import numpy as np


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
