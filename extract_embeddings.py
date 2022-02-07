import os
import glob
import time
import argparse
from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup
import torch
from torchvision.transforms import CenterCrop as CC
from torchvision.transforms import functional as TF
from PIL import ImageFile, Image
from model import *
import train
import sys
import glob
import random
sys.path.append("../")
from tqdm import tqdm

def extract_sketch_embedding(paths, output_name):
    train.args.init_image = paths[0]
    train.args.prompts = ""
    train.args.init_cutn = 32

    # train.args.width = 
    # train.args.height = 

    mh = ModelHost(train.args)
    os.makedirs(f"{args.save_root}/results", exist_ok=True)

    for dataset_size in reversed([1000]):
        # sample 20% of paths
        # paths = random.sample(paths, int(len(paths) * 0.2))
        print(len(paths))
        paths_sampled = random.sample(paths, dataset_size)

        avg_embeddings = []
        batch = []
        for path in tqdm(paths_sampled):
            init_img = mh.load_init_image(path, 400, 400, mh.device)
            cuts = mh.make_cutouts_init(init_img)
            cuts_norm = mh.normalize(cuts.squeeze())
            batch.append(cuts_norm)

            if len(batch) >= (500 // mh.args.init_cutn):
                batch = torch.cat(batch, dim=0)
                embed = mh.perceptor.encode_image(batch).float()
                embed_avg = torch.mean(embed.double(), dim=0)
                avg_embeddings.append((embed_avg.unsqueeze(0)))
                batch = []

        else:
            if batch:
                batch = torch.cat(batch, dim=0)
                embed = mh.perceptor.encode_image(batch).float()
                embed_avg = torch.mean(embed.double(), dim=0)
                avg_embeddings.append((embed_avg.unsqueeze(0)))
                batch = []

            

        avg_embeddings = torch.cat(avg_embeddings, dim=0)
        print(avg_embeddings.shape)
        total_avg = torch.mean(avg_embeddings, dim=0)
        print(total_avg.shape)

        torch.save(total_avg, f"{args.save_root}/results/{output_name}_{dataset_size}.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--path', type=str, default = f"/content/Sketch-Simulator/256x256/sketch/**/**/*.png", help='image path(s).')
    parser.add_argument('--save_root', type=str, default = "/content/Sketch-Simulator/", help='Root directory to save')
    parser.add_argument('--save_name', type=str, default = "sketchy_cutouts_vit-L", help='Root directory to save')
    # parser.add_argument('--padding', type=int, default = 0, help='If to pad images, if so which ratio to add on each side')
    parser.add_argument('--width', type=int, default = 400)
    parser.add_argument('--height', type=int, default = 400)
    args = parser.parse_args()


    path = glob.glob(args.path)
    extract_sketch_embedding(path, args.save_name)