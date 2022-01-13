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
sys.path.append("../")

def get_quickdraw_data(items_per_class, save_root, pad_images):
    d = QuickDrawData(jit_loading=True)

    qd_path = "quickdraw"
    items_per_class = args.items_per_class

    for item in d.drawing_names:
        os.makedirs(f"{args.save_root}/{qd_path}/{'_'.join(item.split(' '))}", exist_ok=True)
        if len(glob.glob(f"{args.save_root}/{qd_path}/{'_'.join(item.split(' '))}/*"))  >= args.items_per_class: continue
        qdg = QuickDrawDataGroup(item, max_drawings=None)
        for i, drawing in enumerate(qdg.drawings):
            drawing.image.save(f"quickdraw/{'_'.join(item.split(' '))}/{i}.png")
            # img = drawing.get_image(stroke_color=(0, 0, 0), stroke_width=2, bg_color=(255, 255, 255))
            if i == args.items_per_class: break
        print(f"{item} done")
    
    if args.pad_images:
        test_path = f"{args.save_root}/{qd_path}/{'_'.join(item.split(' '))}/0.png"
        pil_image = Image.open(test_path).convert('RGB')
        w, h = pil_image.size
        pad_white = CC((w+ w//args.pad_images, h+ h//args.pad_images))
        for item in d.drawing_names:
            for img in glob.glob(f"{args.save_root}/{qd_path}/{'_'.join(item.split(' '))}/*"):
                pil_image = Image.open(img).convert('RGB')
                pil_image = TF.to_tensor(pil_image)
                pil_image = torch.where(pil_image == 0, torch.ones_like(pil_image)*0.0001, pil_image)
                pil_image = pad_white(pil_image)
                pil_image = torch.where(pil_image == 0, torch.ones_like(pil_image), pil_image)
                pil_image = torch.where(pil_image == 0.0001, torch.zeros_like(pil_image), pil_image)
                pil_image = TF.resize(pil_image, (w,h))
                TF.to_pil_image(pil_image.cpu()).save(img)

def extract_sketch_emb_qd(args):
    d = QuickDrawData(jit_loading=True)
    qd_path = "quickdraw"
    mh = ModelHost(train.args)
     
    os.makedirs(f"{args.save_root}/results", exist_ok=True)
    means = []
    for i, item in enumerate(d.drawing_names):
        sketch_paths = glob.glob(f"{args.save_root}/{qd_path}/{'_'.join(item.split(' '))}/*")
        if len(sketch_paths) <= args.items_per_class: continue 
        now = time.time() 
        embeddings = mh.embed_images_cuts(sketch_paths[:args.items_per_class])
        emb_mean = torch.mean(embeddings, dim=0).unsqueeze(0)
        means.append(emb_mean)
    
        if not i % 10:
            print(f"analyzing {item} took {str(time.time()- now)}")
            print(f"Checkpointing ovl_mean")
            ovl_mean = torch.mean(torch.cat(means, dim=0), dim=0).unsqueeze(0)
            torch.save(ovl_mean, f"{args.save_root}/results/ovl_mean_sketch_chpt.pth")   

    ovl_mean = torch.mean(torch.cat(means, dim=0), dim=0).unsqueeze(0)
    torch.save(ovl_mean, f"{args.save_root}/results/ovl_mean_sketch.pth")

def extract_sketch_embedding(paths):
    mh = ModelHost(train.args)
    os.makedirs(f"{args.save_root}/results", exist_ok=True)

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
    parser.add_argument('--path', type=list, default = "", help='image path(s). If empty, will use quickdraw')
    parser.add_argument('--items_per_class', type=int, default = 1000, help='Number of items to analyze per quickdraw class')
    parser.add_argument('--save_root', type=str, default = "", help='Root directory to save')
    parser.add_argument('--pad_images', type=int, default = 0, help='If to pad images, if so which ratio to add on each side')
    parser.add_argument('--width', type=int, default = 400)
    parser.add_argument('--height', type=int, default = 400)
    parser.add_argument('--cutn', type=int, default = 64)
    args = parser.parse_args()

    if not args.path:
        get_quickdraw_data(args.items_per_class, args.save_root, args.pad_images)
        extract_sketch_emb_qd(args.items_per_class, args.save_root)
    else:
        import glob
        path = glob.glob(args.path)
        extract_sketch_embedding(path)