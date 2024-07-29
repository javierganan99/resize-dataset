def hsv_to_bgr(h, s, v):
    """
    Converts an HSV color value to BGR.

    Given a color in the hue, saturation, and value (HSV) color space, this function converts it
    into the blue, green, and red (BGR) color space. The hue should be a value between 0 and 1,
    representing a position on the color wheel. Saturation and value should also be between 0 and 1,
    adjusting the intensity and brightness of the color respectively.

    Args:
        h (float): The hue of the color, must be between 0.0 and 1.0.
        s (float): The saturation of the color, must be between 0.0 and 1.0.
        v (float): The value (brightness) of the color, must be between 0.0 and 1.0.

    Returns:
        tuple: The corresponding BGR color as a tuple with three integers ranging from 0 to 255.
    """
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    switch = {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }
    r, g, b = switch.get(i % 6)
    return (int(b * 255), int(g * 255), int(r * 255))


def generate_n_unique_colors(keys):
    """
    Generates a dictionary of unique colors based on the given keys.

    This function creates a mapping of unique colors to the specified keys using
    the HSV color model. If an integer is provided, it generates colors for that
    many keys. The colors are computed using a golden ratio for even distribution.

    Args:
        keys (int | list): An integer representing the number of unique colors to
            generate or a list of keys for which colors will be generated.

    Returns:
        dict: A dictionary where the keys are the provided input keys and the
            values are the corresponding unique colors in BGR format.
    """
    if isinstance(keys, int):
        n = keys
        keys = list(range(n))
    else:
        n = len(keys)
    hue = 0
    golden_ratio_conjugate = 0.618033988749895
    color_dict = {}
    for key in keys:
        color_dict[key] = hsv_to_bgr(hue, 0.5, 0.95)
        hue += golden_ratio_conjugate
        hue %= 1
    return color_dict
