# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-21
# version: 0.0.1
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, Iterable, Union, Tuple

import torch
import copy
from pathlib import Path

import molpot as mpot
from molpot import keywords as kw

logger = logging.getLogger(__name__)

class DataProxy:

    def __init__(self):

        self._data: Dict[str, torch.Tensor] = {}
        self._frames = []

    def add_frame(self, frame):
        self._frames.append(frame)
    