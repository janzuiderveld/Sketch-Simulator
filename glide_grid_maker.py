import glob
from PIL import Image, ImageOps

selection = glob.glob("/content/drive/MyDrive/AI/sketch-to-image/outputs/glide_outputs_cut_selection/*.png")

print(selection)
path_lookup = dict()
scores = dict()

for sel in selection:
    name, score = sel.split("/")[-1].split(".")[0].split("_")
    score = float(score)
    
    # save for sorting 
    scores[name] = score

    # save for lookup
    path_lookup[name] = sel

# get top 10 best score paths
top_10 = sorted(scores, key=scores.get, reverse=True)[:10]

print(top_10)