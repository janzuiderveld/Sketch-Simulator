
import glob
items = "/content/drive/MyDrive/AI/sketch-to-image/outputs/bullshit/*"
items = glob.glob(items)

for item in items:
    src, style, num = item.split("_")
    