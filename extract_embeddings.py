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
sys.path.append("../")

def extract_sketch_embedding(paths):
    train.args.init_image = paths[0]
    mh = ModelHost(train.args)
    os.makedirs(f"{args.save_root}/results", exist_ok=True)

    for path in paths:
        init_img = mh.load_init_image(path, 400, 400, mh.device)
        batch = mh.make_cutouts_init(init_img)
        # batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = mh.perceptor.encode_image(mh.normalize(batch)).float()

    # paths = paths.split(",")
    # all_items = []
    # for path in paths:
    #     items = glob.glob(f"{path.strip()}/*")
    #     all_items.extend(items)

    embeddings = mh.embed_images_cuts(paths, width = args.width, height = args.height)
    print(embeddings.shape)
    ovl_mean = torch.mean(embeddings, dim=0).unsqueeze(0)
    torch.save(ovl_mean, f"{args.save_root}/results/ovl_mean_sketch_new.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--path', type=str, default = f"/content/drive/MyDrive/AI/sketch-to-image/clip_prototypical/*", help='image path(s).')
    parser.add_argument('--items_per_class', type=int, default = 1000, help='Number of items to analyze per quickdraw class')
    parser.add_argument('--save_root', type=str, default = "", help='Root directory to save')
    parser.add_argument('--padding', type=int, default = 0, help='If to pad images, if so which ratio to add on each side')
    parser.add_argument('--width', type=int, default = 400)
    parser.add_argument('--height', type=int, default = 400)
    parser.add_argument('--cutn', type=int, default = 64)
    args = parser.parse_args()


    path = glob.glob(args.path)
    extract_sketch_embedding(path)