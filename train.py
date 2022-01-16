import os
import random
import argparse
from model import *
from utils import *
import wandb

# TODO
# Make server & client scripts. 

# NOTES
# Dataparallel doesn't speed up training on 1080 GPUs.
# 1 step for synthesizing 400x400 images takes about 0.3 seconds on 1080 GPUs.
# Using prompts with a weight of 0 can be useful for comparing runs in large hyperparameter sweeps.

# A charcoal drawing. 8K HD detailed Wallpaper, digital illustration.
# A photorealistic render [on a white background] in unreal engine, trending on ArtStation
# a beautiful epic wondrous fantasy painting. 8K HD detailed Wallpaper, trending on ArtStation
# Try:
# zislaw bdeksinski
# Ivan Aivazovsky
# Henri Martin
# Thomas Kinkade

# step size: minimal 0.87, 1.5 often works well too
# weight reg: 0, 0.0001 and 0.1 tested, 0.0001 often better
# edge_weight: both 5 and 1 worked well
# cutn 128
# cut_pow 1.5
# step_size 0.12

parser = argparse.ArgumentParser()    
parser.add_argument('--vqgan_model', type=str, default='ImageNet' )
parser.add_argument('--clip_model', type=str, default='ViT-B/32' )
parser.add_argument('--display_freq', type=int, default= 10)
parser.add_argument('--log_edges', type=int, default=0)
parser.add_argument('--max_iterations', type=int, default=50)
parser.add_argument('--seed', type=int, default=-1 )
parser.add_argument('--width', type=int, default= 400 )
parser.add_argument('--height', type=int, default= 400 )
parser.add_argument('--wandb', type=int, default=0)
parser.add_argument('--experiment_name', type=str, default="")

parser.add_argument('--cutn', type=int, default=32 )
parser.add_argument('--cut_pow', type=float, default=0.75)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--step_size', type=float, default=0.87)
parser.add_argument('--ema_val', type=float, default=0.0)
parser.add_argument('--init_weight', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=100)

parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--weight_decouple', type=float, default=1)
parser.add_argument('--rectify', type=float, default=0)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--epsilon', type=float, default=1e-16)
parser.add_argument('--edge_weight', type=int, default= 5)
# parser.add_argument('--sketch_embed_weight', type=int, default= 0)

parser.add_argument('--embedding_avg', type=str, default="/content/Sketch-Simulator/results/ovl_mean_sketch.pth")
# parser.add_argument('--embedding_avg', type=str, default="/content/Sketch-Simulator/results/ovl_mean_small.pth")
# parser.add_argument('--embedding_avg', type=str, default="/content/drive/MyDrive/AI/sketch-to-image/overall_embeddings/ovl_mean_sketchy_vanilla.pt")

parser.add_argument('--target_avg_cuts', type=int, default=1)
parser.add_argument('--target_det_cuts', type=int, default=0)
parser.add_argument('--target_full_img', type=int, default=0)
parser.add_argument('--flavor', type=str, default="cumin", help='"ginger", "cumin", "holywater", "det"')

# parser.add_argument('--reset_img_prompt_every', type=int, default= 0)

# parser.add_argument('--start_image', type=str, default=f"/content/drive/MyDrive/AI/sketch-to-image/clip_prototypical/*")
parser.add_argument('--start_image', type=str, default=f"/content/drive/MyDrive/AI/sketch-to-image/clip_prototypical/crocodilian.png")
# parser.add_argument('--start_image', type=str, default="/content/Sketch-Simulator/test_images/0.png")

parser.add_argument('--padding', type=int, default=100)
# parser.add_argument('--start_image', type=str, default="/content/Sketch-Simulator/test_images/*" )
# parser.add_argument('--prompts', type=str, default="a painting in the style of Salvador Dali" )

parser.add_argument('--prompts', type=str, default="a photorealistic 3D render in Unreal Engine, trending on artstation:1.5" )
# parser.add_argument('--prompts', type=str, default="A charcoal drawing | 8K HD detailed Wallpaper, digital illustration.:0" )

parser.add_argument('--altprompts', type=str, default="" )
parser.add_argument('--noise_prompt_weights', type=list, default=[])

parser.add_argument('--path', type=str, default="")
parser.add_argument('--save_root', type=str, default="")
parser.add_argument('--output_dir', type=str, default="test_outputs")

args = parser.parse_args()

args.image_prompts = args.start_image
args.init_image = args.start_image

model_names={'ImageNet': 'vqgan_imagenet_f16_16384', 'WikiArt': 'wikiart_16384'}
args.vqgan_model = model_names[args.vqgan_model]
args.size = [args.width, args.height]
if args.seed == -1:
    args.seed = None
if args.init_image == "":
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
        if not args.experiment_name:
            experiment_name = wandb.util.generate_id()
        else: experiment_name = args.experiment_name
        run = wandb.init(project="Sketch-to-image", 
                        group=experiment_name)
        config = wandb.config
        config.update(args)
    else:
        config = args

    if "*" in args.start_image:
        import glob
        start_images = glob.glob(args.start_image)
        print(start_images)
        for image in start_images:
            print(image)
            # args.start_image = image
            # args.init_image = image
            # args.image_prompts = [image]
            # wandb.config.update({"allow_val_change":True})  
            wandb.config.update({"start_image": image, 'init_image': image, 'image_prompts': [image]}, allow_val_change=True)  
            print("changed config")
            mh = ModelHost(config)
            mh.run()
    else:
        mh = ModelHost(config)
        mh.run()

if __name__ == "__main__":
    os.makedirs("steps", exist_ok=True)
    Main()
