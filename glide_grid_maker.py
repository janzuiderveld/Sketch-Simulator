import glob
from PIL import Image, ImageOps

selection = glob.glob("/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/*")

print(selection)
path_lookup = dict()
scores = dict()

for sel in selection:
    name, score = sel.split("/")[-1].split("_")
    if name in ["spoon", "teapot",  "bicycle", "chair", "hamburger"]:continue
    print(score)
    score = score[:-4]
    print(score)
    score = float(score)
    
    # save for sorting 
    scores[name] = score

    # save for lookup
    path_lookup[name] = sel

# get top 10 best score paths
top_10 = sorted(scores, key=scores.get, reverse=True)[:10]

print(top_10)

input_images = glob.glob("/content/drive/MyDrive/AI/sketch-to-image/a_complete_clean_and_recognizable_sketch/**")


# make a 5 by 2 grid of the top 10
img_size = Image.open(selection[0]).size
img_width = img_size[0]
img_height = img_size[1]
img_width_per_image = img_width * 2

# make the grid
grid = Image.new("RGB", (img_width_per_image * 2, img_height * 5))

for i, item in enumerate(top_10):
    # get the correct image
    img = Image.open(path_lookup[item])

    # get the corresponding input image
    src_path = [inp for inp in input_images if item in inp]
    src_img = Image.open(src_path[0])
    # reszie the image
    src_img = src_img.resize((img_width, img_height))

    if i < 5:
        # paste src_img left to each image
        grid.paste(src_img, (0, i*img_height))
        # paste img right to each image
        grid.paste(img, (img_width_per_image//2, i*img_height))
    else:
        # paste src_img left to each image
        grid.paste(src_img, (img_width_per_image, (i-5)*img_height))
        # paste img right to each image
        grid.paste(img, (img_width_per_image + img_width_per_image//2, (i-5)*img_height))
        
# save the grid
grid.save(f"/content/drive/MyDrive/AI/sketch-to-image/outputs/glide_grid.png")



    # print(item)
    # print(scores[item])


    # img = Image.open(path_lookup[item])
    # img = img.resize((256, 256))


    # # img.save(f"/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/{item}.png")                               