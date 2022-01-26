
import glob
items = "/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit/*.png"
items = glob.glob(items)


srcs = set()
styles = set()

for item in items:
    print(item)
    src, style, num = item.split("_")
    srcs.add(src)
    styles.add(style)

import matplotlib.pyplot as plt
for i, src in enumerate(srcs):
    for j, style in enumerate(styles):
        plt.subplot(len(srcs), len(styles), i*len(styles) + j + 1)
        plt.imshow(plt.imread(glob.glob(f"/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit/{src}_{style}_*")[0]))
        plt.axis('off')
plt.show()
plt.savefig("/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit/bullshit.png")



# plot