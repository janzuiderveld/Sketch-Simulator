from utils_old import MakeCutoutsDet, save_tensor_as_img

cutter = MakeCutoutsDet(221)

from PIL import Image
import torch   
import numpy as np

#load image
img = Image.open('/content/Sketch-Simulator/test_images/eedb70bc-7a45-41cd-98e1-1f91f6285803.jpeg')
img = torch.from_numpy(np.array(img)).float().unsqueeze(0)

cutter(img)