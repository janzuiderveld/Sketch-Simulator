
import glob
items = "/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit/*.png"
items = glob.glob(items)

items_src = "/content/drive/MyDrive/AI/sketch-to-image/a_complete_clean_and_recognizable_sketch/selection/"
# items_src = glob.glob(items_src)

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
grid_size = (len(styles)+1, len(srcs))
grid = Image.new("RGB", (img_size[0] * grid_size[0], img_size[1] * grid_size[1]))

# first, set items src as the leftmost column
for i, item_src in enumerate(items_src):

for i, src in enumerate(srcs):
    img = Image.open(items_src + src + ".png")
    grid.paste(img, (0, i * img_size[1]))
    for j, style in enumerate(styles):
        item = [item for item in items if src in item and style in item][0]
        img = Image.open(item)
        grid.paste(img, ((j+1) * img_size[0], i * img_size[1]))

# save grid
grid.save("/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit.png")