
import glob
from CLIP import clip
import cv2
from torchvision.transforms import functional as TF
import torch
from collections import defaultdict
from google.colab.patches import cv2_imshow
import random
import os
drive_location = '/content/drive/MyDrive/AI/sketch-to-image'
os.makedirs(f"{drive_location}/clip_prototypical", exist_ok=True)
os.makedirs(f"{drive_location}/overall_embeddings", exist_ok=True)
device = 'cuda'
perceptor = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
input_res = perceptor.visual.input_resolution


def load_image_batch(paths):
  img_batch = []
  for i, item in enumerate(paths):
    # print(item)
    img = cv2.imread(item)
    img = cv2.resize(img, (input_res, input_res))
    img = TF.to_tensor(img).to(device).unsqueeze(0) * 2 - 1
    img_batch.append(img)
  img_batch = torch.cat(img_batch, dim = 0)
  return img_batch

def encode_image_batch(paths):
  img_batch = load_image_batch(paths)
  img_embed = perceptor.encode_image(img_batch)
  return img_embed


  ### DATA LOOKUP BY CLIP ###

# total_avg = torch.load(f"{drive_location}/overall_embeddings/ovl_mean_sketchy_vanilla.pt")
total_avg = torch.load(f"/content/Sketch-Simulator/results/ovl_mean_sketchy_cutouts.pt")

dir = "glide_outputs_cut_selection"

# dir = "test"
os.makedirs(f"{drive_location}/{dir}", exist_ok=True)

analyzed_classes = []
# # sketch_classes = glob.glob(f"{drive_location}/sketchy/sketch/**/*")

sketch_classes = glob.glob(f"{drive_location}/glide_outputs_cut/*0*")



for sketch_class in sketch_classes:
  name = sketch_class.split("/")[-1].split("_")[0]
  if name in analyzed_classes: continue
  descr = f"a photo of a {name}, a clear depiction of a {name}"
  # descr = f"a clear depiction of a {name}"
  # descr = f"a photorealistic 3D render of a {name} in Unreal Engine"
  text_embed = perceptor.encode_text(clip.tokenize(descr).to(device))
  text_embed = text_embed / torch.linalg.norm(text_embed, dim=1)

  # get "{drive_location}/glide_outputs_cut/{name}_[1-10].png" with [1-10] replaced with appropriate regex
  class_imgs = glob.glob(f"{drive_location}/glide_outputs_cut/{name}*.png")[1:]

  img_batch = load_image_batch(class_imgs)

  img_embed = perceptor.encode_image(img_batch) 
  img_embed = img_embed / torch.linalg.norm(text_embed, dim=1)
  
  # text: 1 x 512 @ img.T: 512, 32
  print(text_embed.shape, img_embed.shape)
  scores = text_embed @ img_embed.T
  best_score_idx = torch.argmax(scores, dim=1).detach().cpu().squeeze().numpy()
  scores = scores.detach().cpu().squeeze().numpy()
  
  # grab best score as str
  best_score = scores[best_score_idx]

  print(descr)
  img = cv2.imread(class_imgs[best_score_idx])
  fname = f"{name}_{str(best_score)}.png"
  cv2.imwrite(f"{drive_location}/{dir}/{fname}", img)
  analyzed_classes.append(name)
  torch.cuda.empty_cache()
