# https://github.com/xdit-project/xDiT/blob/1c31746e2f903e791bc2a41a0bc23614958e46cd/comfyui-xdit/utils.py

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

def convert_images_to_tensors(images: list[Image.Image]):
    return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])
