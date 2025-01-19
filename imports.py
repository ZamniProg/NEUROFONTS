import os
import json
import shutil
from tqdm import tqdm

import datasets
from datasets import DatasetDict

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from PIL import Image
from pycocotools.coco import COCO

from typing import Union, List, Tuple, Dict, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.ops