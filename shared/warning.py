# ------------------------------------------
# FUNCTIONS for WARNING MESSAGE
# ------------------------------------------

import numpy as np


def check_img_supported(img):
    """
    Check if img is supported integer image type. Raise type error if not.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Only integer image types are supported. Got %s." % type(img))


def check_input_supported(inp, supported):
    """
    Check if input value is in supported value range.  Raise value error if not.

    :param inp: All, input value
    :param supported: list, supported input range
    """
    if not inp in supported:
        raise ValueError("Only accepts %s. Got %s" % (supported, inp))
