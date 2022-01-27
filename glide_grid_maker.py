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

for item in top_10:
    print(item)
    print(scores[name])
#     img = Image.open(path_lookup[item])
#     img = img.resize((256, 256))
#     img.save(f"/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/{item}.png")                               