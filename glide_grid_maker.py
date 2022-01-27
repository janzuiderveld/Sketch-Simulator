import glob
from PIL import Image, ImageOps

import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

# Sampling parameters
prompt = ""
batch_size = 1
guidance_scale = 3.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997


selection = glob.glob("/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/*")

print(selection)
path_lookup = dict()
scores = dict()

for sel in selection:
    name, score = sel.split("/")[-1].split("_")
    if name in ["spoon", "teapot",  "bicycle", "chair", "hamburger", "rabbit", "rifle", "cat", "table"]:continue
    print(score)
    score = score[:-4]
    print(score)
    score = float(score)
    
    # save for sorting 
    scores[name] = score

    # save for lookup
    path_lookup[name] = sel

# get top 10 best score paths
top_10 = sorted(scores, key=scores.get, reverse=True)[:12]

print(top_10)

input_images = glob.glob("/content/drive/MyDrive/AI/sketch-to-image/a_complete_clean_and_recognizable_sketch/**")


# make a 3 by 4 grid of the top 12
img_size = Image.open(selection[0]).size
img_width = img_size[0]
img_height = img_size[1]
img_width_per_image = img_width * 2

# make the grid
grid = Image.new("RGB", (img_width_per_image * 4, img_height * 3))

for i, item in enumerate(top_10):
    # get the correct image
    img = Image.open(path_lookup[item])

    # get the corresponding input image
    src_path = [inp for inp in input_images if item in inp]
    src_img = Image.open(src_path[0])
    # reszie the image
    src_img = src_img.resize((img_width, img_height))

    if i < 3:
        # paste src_img left to each image
        grid.paste(src_img, (0, i*img_height))
        # paste img right to each image
        grid.paste(img, (img_width_per_image//2, i*img_height))

    elif i < 6:
        # paste src_img left to each image
        grid.paste(src_img, (img_width_per_image, (i-3)*img_height))
        # paste img right to each image
        grid.paste(img, (img_width_per_image + img_width_per_image//2, (i-3)*img_height))
    elif i < 9:
        # paste src_img left to each image
        grid.paste(src_img, (img_width_per_image*2, (i-6)*img_height))
        # paste img right to each image
        grid.paste(img, (img_width_per_image*2 + img_width_per_image//2, (i-6)*img_height))
    else:
        # paste src_img left to each image
        grid.paste(src_img, (img_width_per_image*3, (i-9)*img_height))
        # paste img right to each image
        grid.paste(img, (img_width_per_image*3 + img_width_per_image//2, (i-9)*img_height))
        
# save the grid
grid.save(f"/content/drive/MyDrive/AI/sketch-to-image/outputs/glide_grid.png")



    # print(item)
    # print(scores[item])


    # img = Image.open(path_lookup[item])
    # img = img.resize((256, 256))


    # # img.save(f"/content/drive/MyDrive/AI/sketch-to-image/glide_outputs_cut_selection/{item}.png")                               