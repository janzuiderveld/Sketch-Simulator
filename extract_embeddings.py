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

def extract_sketch_embedding(paths):
    train.args.init_image = paths[0]
    train.args.init_cutn = 32
    mh = ModelHost(train.args)
    os.makedirs(f"{args.save_root}/results", exist_ok=True)

    # sample 20% of paths
    paths = random.sample(paths, int(len(paths) * 0.2))

    avg_embeddings = []
    batch = []
    for path in tqdm(paths):
        init_img = mh.load_init_image(path, 400, 400, mh.device)
        cuts = mh.make_cutouts_init(init_img)
        cuts_norm = mh.normalize(cuts)
        batch.append(cuts_norm)

        if len(batch) >= (5000 // 32):
            batch = torch.cat(batch, dim=0)
            # input(batch.shape)
           # batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = mh.perceptor.encode_image(batch).float()
            # input(embed.shape)
            embed_avg = torch.mean(embed.double(), dim=0)
            # input(embed_avg.shape)
            avg_embeddings.append((embed_avg.unsqueeze(0)))
            batch = []
        

    avg_embeddings = torch.cat(avg_embeddings, dim=0)
    print(avg_embeddings.shape)
    total_avg = torch.mean(avg_embeddings, dim=0)
    print(total_avg.shape)

    torch.save(total_avg, f"{args.save_root}/results/ovl_mean_sketchy_photo_cutouts.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    # parser.add_argument('--path', type=str, default = f"/content/Sketch-Simulator/256x256/sketch/**/**/*.png", help='image path(s).')
    parser.add_argument('--path', type=str, default = f"/content/Sketch-Simulator/256x256/photo/**/**/*.jpg", help='image path(s).')
    parser.add_argument('--items_per_class', type=int, default = 1000, help='Number of items to analyze per quickdraw class')
    parser.add_argument('--save_root', type=str, default = "/content/Sketch-Simulator/", help='Root directory to save')
    parser.add_argument('--padding', type=int, default = 0, help='If to pad images, if so which ratio to add on each side')
    parser.add_argument('--width', type=int, default = 400)
    parser.add_argument('--height', type=int, default = 400)
    parser.add_argument('--cutn', type=int, default = 64)
    args = parser.parse_args()


    path = glob.glob(args.path)
    extract_sketch_embedding(path)