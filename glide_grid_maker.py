import glob
from PIL import Image, ImageOps

selection = glob.glob("/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/*")

print(selection)
path_lookup = dict()
scores = dict()

for sel in selection:
    name, score = sel.split("/")[-1].split("_")
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
print(input_images)

# make a 2 by 5 grid of the top 10
img_size = Image.open(selection[0]).size




for item in top_10:
    print(item)
    print(scores[item])


    img = Image.open(path_lookup[item])
    img = img.resize((256, 256))


    # img.save(f"/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/{item}.png")                               