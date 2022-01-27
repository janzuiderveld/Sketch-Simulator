import glob
from PIL import Image, ImageOps

# folder_name = "dataset_size_experiment"
folder_name = "glide_outputs"
items = f"/content/drive/MyDrive/AI/sketch-to-image/{folder_name}/*.png"
items = glob.glob(items)



# items_src = "/content/drive/MyDrive/AI/sketch-to-image/a_complete_clean_and_recognizable_sketch/selection2/"


items_src = "/content/Sketch-Simulator/256x256/sketch/**/**/*.png"
items_src = glob.glob(items_src)


