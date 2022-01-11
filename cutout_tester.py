from utils_old import MakeCutoutsDet

cutter = MakeCutoutsDet(221)

from PIL import Image
#load image
img = Image.open('/content/Sketch-Simulator/test_images/eedb70bc-7a45-41cd-98e1-1f91f6285803.jpeg')

cutter(img)