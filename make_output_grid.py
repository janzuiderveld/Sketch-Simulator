import glob
from PIL import Image, ImageOps

folder_name = "dataset_size_experiment"
items = f"/content/drive/MyDrive/AI/sketch-to-image/outputs/{folder_name}/*.png"
items = glob.glob(items)

items_src = "/content/drive/MyDrive/AI/sketch-to-image/a_complete_clean_and_recognizable_sketch/selection2/"
# items_src = glob.glob(items_src)

srcs = set()
styles = set()
avgs = set()

for item in items:
    # print(item)
    src, avg_emb, style, num = item.split("/")[-1].split("_")
    srcs.add(src)
    styles.add(style)
    avgs.add(avg_emb)

avgs = list(avgs)
avgs = (list(reversed(list(sorted(avgs)))))
avgs = avgs[2:-1]

# fill a grid with images, style on x-axis, src on y-axis
img_size = Image.open(items[0]).size

# grid_size = (len(styles)+1, len(srcs))
grid_size = (len(avgs)+1, len(srcs))

grid = Image.new("RGB", (img_size[0] * grid_size[0], img_size[1] * grid_size[1]))


print(avgs)

for i, src in enumerate(srcs):
    src_path = items_src + src + "*.png"
    src_path = glob.glob(src_path)
    img = Image.open(src_path[0])
    # resize to fit
    img = img.resize(img_size)
    img = ImageOps.expand(img, border=70, fill=(255, 255, 255))
    img = img.resize(img_size)

    grid.paste(img, (0, i * img_size[1]))
    for j, style in enumerate(avgs):
    # for j, style in enumerate(styles):
        

        item = [item for item in items if src in item and style in item][0]
        img = Image.open(item)
        grid.paste(img, ((j+1) * img_size[0], i * img_size[1]))

# save grid
grid.save(f"/content/drive/MyDrive/AI/sketch-to-image/outputs/{folder_name}.png")