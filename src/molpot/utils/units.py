import re
from typing import Union, Dict

import numpy as np
import pint

__u = pint.UnitRegistry()

def convert(src: tuple[float], tgt_unit: str) -> float:
    """
    Convert a value from one unit to another.

    Args:
        src: The value to convert.
        tgt_unit: The target unit.

    Returns:
        The converted value.
    """
    src_unit = src[1]
    src_value = src[0]
    return (src_value * __u(src_unit)).to(tgt_unit).magnitude