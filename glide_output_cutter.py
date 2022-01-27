import glob, os
from PIL import Image, ImageOps

os.makedirs("thrash", exist_ok=True)

# folder_name = "dataset_size_experiment"
folder_name = "glide_outputs"
items = f"/content/drive/MyDrive/AI/sketch-to-image/{folder_name}/*.png"
items = glob.glob(items)

os.makedirs("/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut", exist_ok=True)

for item in items:
    # open image
    img = Image.open(item)

    # these images consist of 11 vertically stacked images, so we need to cut them up
    # into 11 separate images
    img_size = img.size
    img_width = img_size[0]
    img_height = img_size[1]
    img_width_per_image = img_width // 11
    img_height_per_image = img_height

    # cut up the image into 11 images
    for i in range(11):
        # get the correct image
        img_to_cut = img.crop((i*img_width_per_image, 0, (i+1)*img_width_per_image, img_height_per_image))

        # grab name
        name = item.split("/")[-1].split(".")[0].split("_")[2]

        # save it
        img_to_cut.save(f"/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut/{name}_{i}.png")
# items_src = "/content/drive/MyDrive/AI/sketch-to-image/a_complete_clean_and_recognizable_sketch/selection2/"


items_src = "/content/Sketch-Simulator/256x256/sketch/**/**/*.png"
items_src = glob.glob(items_src)


