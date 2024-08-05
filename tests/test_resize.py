import pytest
import numpy as np
from resize_dataset.image import RESIZE_METHODS


def test_resize_methods():
    for _, resize_method in RESIZE_METHODS.items():
        resize_image_bicubic_scale(resize_method)
        resize_image_bicubic_shape(resize_method)
        edge_cases(resize_method)
        invalid_inputs(resize_method)


def create_test_image(height=10, width=10, channels=3):
    return np.ones((height, width, channels), dtype=np.uint8) * 255


def resize_image_bicubic_scale(resize_method):
    image = create_test_image()
    result = resize_method(image, scale=2)
    assert result.shape == (20, 20, 3)  # New size should be 2x2 of original
    result = resize_method(image, scale=0.5)
    assert result.shape == (5, 5, 3)  # New size should be 0.5x0.5 of original
    result = resize_method(image, scale=1)
    assert result.shape == (10, 10, 3)


def resize_image_bicubic_shape(resize_method):
    image = create_test_image()
    result = resize_method(image, shape=(20, 20))
    assert result.shape == (20, 20, 3)
    result = resize_method(image, shape=(5, 5))
    assert result.shape == (5, 5, 3)


def invalid_inputs(resize_method):
    image = create_test_image()
    with pytest.raises(ValueError, match="Image parameter missing!"):
        resize_method()
    with pytest.raises(TypeError, match="The image should be a numpy ndarray."):
        resize_method("not_an_image", scale=2)
    with pytest.raises(TypeError, match="Scale must be an int or float."):
        resize_method(image, scale="invalid_scale")
    with pytest.raises(TypeError, match="Shape must be a tuple or list."):
        resize_method(image, shape="invalid_shape")


def edge_cases(resize_method):
    image = create_test_image()
    result = resize_method(image, shape=(100, 100))
    assert result.shape == (100, 100, 3)
    result = resize_method(image, scale=4)
    assert result.shape == (image.shape[0] * 4, image.shape[1] * 4, 3)


# Run the tests
if __name__ == "__main__":
    pytest.main()
