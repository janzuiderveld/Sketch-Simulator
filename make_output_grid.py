
import glob
items = "/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit/*.png"
items = glob.glob(items)


srcs = set()
styles = set()

for item in items:
    print(item)
    src, style, num = item.split("/")[-1].split("_")
    srcs.add(src)
    styles.add(style)

from PIL import Image
# fill a grid with images, style on x-axis, src on y-axis
img_size = Image.open(items[0]).size
grid_size = (len(styles), len(srcs))
grid = Image.new("RGB", (img_size[0] * grid_size[0], img_size[1] * grid_size[1]))

for i, src in enumerate(srcs):
    for j, style in enumerate(styles):
        item = [item for item in items if src in item and style in item][0]
        img = Image.open(item)
        grid.paste(img, (j * img_size[0], i * img_size[1]))

# save grid
grid.save("/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit.png")
