import argparse
import math
from pathlib import Path
import sys

sys.path.append('./taming-transformers')
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import CenterCrop as CC
from tqdm.notebook import tqdm

from CLIP import clip
import kornia
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import hashlib
from PIL.PngImagePlugin import PngImageFile, PngInfo
import json

import IPython
from IPython.display import Markdown, display, clear_output
from IPython.display import Image as ImageViewer
import urllib.request
import random

from utils_old import *
from model_old import *

import os
from google.colab import drive

# The Augmentation sequence is included in this code block, but if you want to make changes, you'll need to edit it yourself instead of through the user interface! There's only so much markdown can do.

# A photorealistic render in unreal engine on a white background, trending on ArtStation

# `prompts` is the list of prompts to give to the AI, separated by `|`. With more than one, it will attempt to mix them together.
#prompts = ""
prompts = "8K HD detailed Wallpaper, digital illustration, artstation" #@param {type:"string"}

width =  400#@param {type:"number"}
height =  400#@param {type:"number"}
model = 'ImageNet 16384' #@param ['ImageNet 16384', 'ImageNet 1024', 'WikiArt 1024', 'WikiArt 16384', 'COCO-Stuff', 'FacesHQ', 'S-FLCKR']
# Only the prompts, width, height and model work on pixel art.
Pixel_Art = False #@param {type:"boolean"}
# The flavor effects the output greatly. Each has it's own characteristics and depending on what you choose, you'll get a widely different result with the same prompt and seed. Ginger is the default, nothing special. Cumin results more of a painting, while Holywater makes everythng super funky and/or colorful. Custom is a custom flavor, use the utilities above.
flavor = 'cumin' #@param ["ginger", "cumin", "holywater", "custom"]

# ---

# `folder_name` is the name of the folder you want to output your result(s) to. Previous outputs will NOT be overwritten. By default, it will be saved to the colab's root folder, but the `save_to_drive` checkbox will save it to `MyDrive\VQGAN_Output` instead.
folder_name = "Output"#@param {type:"string"}
save_to_drive = False #@param {type:"boolean"}


# Advanced values. Values of cut_pow below 1 prioritize structure over detail, and vice versa for above 1. Step_size affects how wild the change between iterations is, and if final_step_size is not 0, step_size will interpolate towards it over time.
# Cutn affects on 'Creativity': less cutout will lead to more random/creative results, sometimes barely readable, while higher values (90+) lead to very stable, photo-like outputs
cutn = 128#@param {type:"number"}
cut_pow = 1.5#@param {type:"number"}
# Step_size is like weirdness. Lower: more accurate/realistic, slower; Higher: less accurate/more funky, faster.
step_size = 0.12#@param {type:"number"}
final_step_size = 0.05#@param {type:"number"} 
if final_step_size <= 0: final_step_size = step_size

# ---

# EMA maintains a moving average of trained parameters. The number below is the rate of decay (higher means slower).
ema_val = 0.98 #@param {type:"number"}


input_image = "/content/Sketch-Simulator/test_images/eedb70bc-7a45-41cd-98e1-1f91f6285803.jpeg" #@param {type:"string"}
# input_image = "/content/quickdraw/bird/4.png"

# To use initial or target images, upload it on the left in the file browser. You can also use previous outputs by putting its path below, e.g. `batch_01/0.png`. If your previous output is saved to drive, you can use the checkbox so you don't have to type the whole path.
init_image = input_image
init_image_in_drive = False #@param {type:"boolean"}
transparent_png = False #@param {type:"boolean"}
if init_image_in_drive and init_image:
    init_image = '/content/drive/MyDrive/VQGAN_Output/' + init_image

# Target images work like prompts, and you can provide more than one by separating the filenames with `|`.
target_images = input_image
seed = -1#@param {type:"number"}
images_interval =  10#@param {type:"number"}

# max_iterations excludes iterations spent during the mse phase, if it is being used.
max_iterations = 2000#@param {type:"number"}
batch_size =  1#@param {type:"number"}

# ---

## MSE Regulization. 
#Based off of this notebook: https://colab.research.google.com/drive/1gFn9u3oPOgsNzJWEFmdK-N9h_y65b8fj?usp=sharing - already in credits
use_mse = True #@param {type:"boolean"}
mse_images_interval = images_interval
mse_init_weight =  0.1#@param {type:"number"}
mse_decay_rate =  100#@param {type:"number"}
mse_epoches =  1000#@param {type:"number"}
mse_with_zeros = False #@param {type:"boolean"}

# ---

# Overwrites the usual values during the mse phase if included. If any value is 0, its normal counterpart is used instead.
mse_step_size = 0.87 #@param {type:"number"}
mse_cutn =  32#@param {type:"number"}
mse_cut_pow = 0.75 #@param {type:"number"}


# `altprompts` is a set of prompts that take in a different augmentation pipeline, and can have their own cut_pow. At the moment, the default "alt augment" settings flip the picture cutouts upside down before evaluating. This can be good for optical illusion images. If either cut_pow value is 0, it will use the same value as the normal prompts.
altprompts = "" #@param {type:"string"}
alt_cut_pow = 1.5 #@param {type:"number"}
alt_mse_cut_pow =  0.75#@param {type:"number"}

model_names={'ImageNet 16384': 'vqgan_imagenet_f16_16384', 'ImageNet 1024': 'vqgan_imagenet_f16_1024', 'WikiArt 1024': 'wikiart_1024', 'WikiArt 16384': 'wikiart_16384', 'COCO-Stuff': 'coco', 'FacesHQ': 'faceshq', 'S-FLCKR': 'sflckr'}

if Pixel_Art:
  # Simple setup
  import clipit

  clipit.reset_settings()
  clipit.add_settings(prompts=prompts, vqgan_model=model_names[model], size=[width, height]) #Aspect can be either "widescreen" or "square" 
  clipit.add_settings(quality="best", scale=2.5)
  clipit.add_settings(use_pixeldraw=True) #Doesn't have to be True, but highly recommended
  clipit.add_settings(iterations=max_iterations, display_every=images_interval)

  settings = clipit.apply_settings()
  clipit.do_init(settings)
  clipit.do_run(settings)
else:
  mse_decay = 0

  if use_mse == False:
      mse_init_weight = 0.
  else:
      mse_decay = mse_init_weight / mse_epoches
    
  if os.path.isdir('/content/drive') == False:
      if save_to_drive == True or init_image_in_drive == True:
          drive.mount('/content/drive')

  if seed == -1:
      seed = None
  if init_image == "None":
      init_image = None
  if target_images == "None" or not target_images:
      target_images = []
  else:
      target_images = target_images.split("|")
      target_images = [image.strip() for image in target_images]

  prompts = [phrase.strip() for phrase in prompts.split("|")]
  if prompts == ['']:
      prompts = []

  altprompts = [phrase.strip() for phrase in altprompts.split("|")]
  if altprompts == ['']:
      altprompts = []

  if mse_images_interval == 0: mse_images_interval = images_interval
  if mse_step_size == 0: mse_step_size = step_size
  if mse_cutn == 0: mse_cutn = cutn
  if mse_cut_pow == 0: mse_cut_pow = cut_pow
  if alt_cut_pow == 0: alt_cut_pow = cut_pow
  if alt_mse_cut_pow == 0: alt_mse_cut_pow = mse_cut_pow

  augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomGaussianBlur((3,3),(4.5,4.5),p=0.3),
            #K.RandomGaussianNoise(p=0.5),
            #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
            K.RandomPerspective(0.2,p=0.4, ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),)

  altaugs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=1),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomGaussianBlur((3,3),(4.5,4.5),p=0.3),
            #K.RandomGaussianNoise(p=0.5),
            #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
            K.RandomPerspective(0.2,p=0.4, ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),)

  args = argparse.Namespace(
      prompts=prompts,
      altprompts=altprompts,
      image_prompts=target_images,
      noise_prompt_seeds=[],
      noise_prompt_weights=[],
      size=[width, height],
      init_image=init_image,
      png=transparent_png,
      init_weight= mse_init_weight,
      clip_model='ViT-B/32',
      vqgan_model=model_names[model],
      step_size=step_size,
      final_step_size = final_step_size,
      cutn=cutn,
      cut_pow=cut_pow,
      mse_cutn = mse_cutn,
      mse_cut_pow = mse_cut_pow,
      mse_step_size = mse_step_size,
      display_freq=images_interval,
      mse_display_freq = mse_images_interval,
      max_iterations=max_iterations,
      mse_end = mse_decay_rate * mse_epoches,
      seed=seed,
      folder_name=folder_name,
      save_to_drive=save_to_drive,
      mse_decay_rate = mse_decay_rate,
      mse_decay = mse_decay,
      mse_with_zeros = mse_with_zeros,
      ema_val = 0.98,
      augs = augs,
      altaugs = altaugs,
      alt_cut_pow = alt_cut_pow,
      alt_mse_cut_pow = alt_mse_cut_pow,
  )

  if save_to_drive == True:
      if os.path.isdir('/content/drive/MyDrive/VQGAN_Output/'+folder_name) == False:
          os.makedirs('/content/drive/MyDrive/VQGAN_Output/'+folder_name)
  elif os.path.isdir('/content/'+folder_name) == False:
      os.makedirs('/content/'+folder_name)


  # mh = ModelHost(args)
  # mh.setup_model()
  # embedding = mh.embed_images_cuts(["/content/catface.png", "/content/catface.png", "/content/catface.png"])

  # print(embedding.shape)
  ovl_mean = torch.load("/content/Sketch-Simulator/results/ovl_mean_sketch.pth")


  mh = ModelHost(args)
  x= 0

  for x in range(batch_size):
      mh.setup_model()
      last_iter = mh.run(x)
      x=x+1

  if batch_size != 1:
    #clear_output()
    print("===============================================================================")
    q = 0
    while q < batch_size:
      display(Image('/content/' + folder_name + "/" + str(q) + '.png'))
      print("Image" + str(q) + '.png')
      q += 1