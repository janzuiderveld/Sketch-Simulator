import os
import random
import argparse
from model import *
from utils import *
import wandb

# A charcoal drawing. 8K HD detailed Wallpaper, digital illustration.
# A photorealistic render [on a white background] in unreal engine, trending on ArtStation
# a beautiful epic wondrous fantasy painting. 8K HD detailed Wallpaper, trending on ArtStation

# Try:
# zislaw bdeksinski
# Ivan Aivazovsky
# Henri Martin
# Thomas Kinkade

# TODO
# Make server & client scripts. 

# EXPERIMENTAL
# Try adding reinterpretation of image as loss? Minus the sketch embedding again?

# NOTES
# step size: minimal 0.87, 1.5 often works well too
# weight reg: 0, 0.0001 and 0.1 tested, 0.0001 often better
# edge_weight: both 5 and 1 worked well



parser = argparse.ArgumentParser()    
parser.add_argument('--vqgan_model', type=str, default='ImageNet' )
parser.add_argument('--clip_model', type=str, default='ViT-B/32' )
parser.add_argument('--images_interval', type=int, default= 10 )
parser.add_argument('--log_edges', type=int, default=0)
parser.add_argument('--start_image', type=str, default="quickdraw/bird/4.png" )
parser.add_argument('--max_iterations', type=int, default=30 )
parser.add_argument('--seed', type=int, default=-1 )
parser.add_argument('--width', type=int, default= 400 )
parser.add_argument('--height', type=int, default= 400 )
parser.add_argument('--wandb', type=int, default=1)
    
# parser.add_argument('--cutn', type=int, default=128 )
# parser.add_argument('--cut_pow', type=float, default=1.5 )
# parser.add_argument('--step_size', type=float, default=0.12 )

parser.add_argument('--cutn', type=int, default=32 )
parser.add_argument('--cut_pow', type=float, default=0.75)
parser.add_argument('--flavor', type=str, default='cumin'  )

parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--step_size', type=float, default=0.87)
parser.add_argument('--weight_reg', type=float, default=0)
parser.add_argument('--weight_decouple', type=float, default=1)
parser.add_argument('--rectify', type=float, default=0)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--epsilon', type=float, default=1e-16)

parser.add_argument('--prompts', type=str, default="A charcoal drawing | 8K HD detailed Wallpaper, digital illustration." )
# parser.add_argument('--prompts', type=str, default="A photorealistic render in unreal engine | a white background | trending on ArtStation" )

parser.add_argument('--edge_weight', type=int, default= 5)
parser.add_argument('--sketch_embed_weight', type=int, default= 0)
parser.add_argument('--reset_img_prompt_every', type=int, default= 0)

args = parser.parse_args()

args.image_prompts = args.start_image
args.init_image = args.start_image

model_names={'ImageNet': 'vqgan_imagenet_f16_16384', 'WikiArt': 'wikiart_16384'}
args.vqgan_model = model_names[args.vqgan_model]
args.size = [args.width, args.height]
if args.seed == -1:
    args.seed = None
if args.init_image == "None":
    args.init_image = None
if args.image_prompts == "None" or not args.image_prompts:
    args.image_prompts = []
else:
    args.image_prompts = args.image_prompts.split("|")
    args.image_prompts = [image.strip() for image in args.image_prompts]
args.prompts = [phrase.strip() for phrase in args.prompts.split("|")]
if args.prompts == ['']:
    args.prompts = []

def Main():
    print(args.prompts)
    if args.wandb:
        experiment_name = wandb.util.generate_id()
        args.experiment_name = experiment_name
        run = wandb.init(project="Sketch-sim", 
                        group=experiment_name)
        config = wandb.config
        config.update(args)
    else:
        config = args

    mh = ModelHost(config)
    mh.train()

if __name__ == "__main__":
    Main()
