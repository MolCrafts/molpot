import re
from typing import Union, Dict

import numpy as np
import pint

__u = pint.UnitRegistry()

def convert(value:float, src_unit: str, tgt_unit: str) -> float:
    """
    Convert a value from one unit to another.

    Args:
        src: The value to convert.
        tgt_unit: The target unit.

    Returns:
        The converted value.
    """
    return (value * __u(src_unit)).to(tgt_unit).magnitude