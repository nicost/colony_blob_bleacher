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
