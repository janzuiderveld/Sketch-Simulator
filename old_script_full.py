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

# def sinc(x):
#     return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


# def lanczos(x, a):
#     cond = torch.logical_and(-a < x, x < a)
#     out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
#     return out / out.sum()


# def ramp(ratio, width):
#     n = math.ceil(width / ratio + 1)
#     out = torch.empty([n])
#     cur = 0
#     for i in range(out.shape[0]):
#         out[i] = cur
#         cur += ratio
#     return torch.cat([-out[1:].flip([0]), out])[1:-1]


# def resample(input, size, align_corners=True):
#     n, c, h, w = input.shape
#     dh, dw = size

#     input = input.view([n * c, 1, h, w])

#     if dh < h:
#         kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
#         pad_h = (kernel_h.shape[0] - 1) // 2
#         input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
#         input = F.conv2d(input, kernel_h[None, None, :, None])

#     if dw < w:
#         kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
#         pad_w = (kernel_w.shape[0] - 1) // 2
#         input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
#         input = F.conv2d(input, kernel_w[None, None, None, :])

#     input = input.view([n, c, h, w])
#     return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

# def lerp(a, b, f):
#     return (a * (1.0 - f)) + (b * f);

# class ReplaceGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x_forward, x_backward):
#         ctx.shape = x_backward.shape
#         return x_forward

#     @staticmethod
#     def backward(ctx, grad_in):
#         return None, grad_in.sum_to_size(ctx.shape)


# replace_grad = ReplaceGrad.apply


# class ClampWithGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, min, max):
#         ctx.min = min
#         ctx.max = max
#         ctx.save_for_backward(input)
#         return input.clamp(min, max)

#     @staticmethod
#     def backward(ctx, grad_in):
#         input, = ctx.saved_tensors
#         return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


# clamp_with_grad = ClampWithGrad.apply


# def vector_quantize(x, codebook):
#     d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
#     indices = d.argmin(-1)
#     x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
#     return replace_grad(x_q, x)


# class Prompt(nn.Module):
#     def __init__(self, embed, weight=1., stop=float('-inf')):
#         super().__init__()
#         self.register_buffer('embed', embed)
#         self.register_buffer('weight', torch.as_tensor(weight))
#         self.register_buffer('stop', torch.as_tensor(stop))

#     def forward(self, input):
#         input_normed = F.normalize(input.unsqueeze(1), dim=2)
#         embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
#         dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
#         dists = dists * self.weight.sign()
#         return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


# def parse_prompt(prompt):
#     vals = prompt.rsplit(':', 2)
#     vals = vals + ['', '1', '-inf'][len(vals):]
#     return vals[0], float(vals[1]), float(vals[2])

# class EMATensor(nn.Module):
#     """implmeneted by Katherine Crowson"""
#     def __init__(self, tensor, decay):
#         super().__init__()
#         self.tensor = nn.Parameter(tensor)
#         self.register_buffer('biased', torch.zeros_like(tensor))
#         self.register_buffer('average', torch.zeros_like(tensor))
#         self.decay = decay
#         self.register_buffer('accum', torch.tensor(1.))
#         self.update()
    
#     @torch.no_grad()
#     def update(self):
#         if not self.training:
#             raise RuntimeError('update() should only be called during training')

#         self.accum *= self.decay
#         self.biased.mul_(self.decay)
#         self.biased.add_((1 - self.decay) * self.tensor)
#         self.average.copy_(self.biased)
#         self.average.div_(1 - self.accum)

#     def forward(self):
#         if self.training:
#             return self.tensor
#         return self.average


############################################################################################
############################################################################################


# class MakeCutoutsCustom(nn.Module):
#     def __init__(self, cut_size, cutn, cut_pow, augs):
#         super().__init__()
#         self.cut_size = cut_size
#         print(f'cut size: {self.cut_size}')
#         self.cutn = cutn
#         self.cut_pow = cut_pow
#         self.noise_fac = 0.1
#         self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
#         self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
#         self.augs = augs
        
#         nn.Sequential(
#           K.RandomHorizontalFlip(p=Random_Horizontal_Flip),
#           K.RandomSharpness(Random_Sharpness,p=Random_Sharpness_P),
#           K.RandomGaussianBlur((Random_Gaussian_Blur),(Random_Gaussian_Blur_W,Random_Gaussian_Blur_W),p=Random_Gaussian_Blur_P),
#           K.RandomGaussianNoise(p=Random_Gaussian_Noise_P),
#           K.RandomElasticTransform(kernel_size=(Random_Elastic_Transform_Kernel_Size_W, Random_Elastic_Transform_Kernel_Size_H), sigma=(Random_Elastic_Transform_Sigma), p=Random_Elastic_Transform_P),
#           K.RandomAffine(degrees=Random_Affine_Degrees, translate=Random_Affine_Translate, p=Random_Affine_P, padding_mode='border'),
#           K.RandomPerspective(Random_Perspective,p=Random_Perspective_P),
#           K.ColorJitter(hue=Color_Jitter_Hue, saturation=Color_Jitter_Saturation, p=Color_Jitter_P),)
#           #K.RandomErasing((0.1, 0.7), (0.3, 1/0.4), same_on_batch=True, p=0.2),)

#     def set_cut_pow(self, cut_pow):
#       self.cut_pow = cut_pow
    
#     def forward(self, input):
#         sideY, sideX = input.shape[2:4]
#         max_size = min(sideX, sideY)
#         min_size = min(sideX, sideY, self.cut_size)
#         cutouts = []
#         cutouts_full = []
#         noise_fac = 0.1
        
        
#         min_size_width = min(sideX, sideY)
#         lower_bound = float(self.cut_size/min_size_width)
        
#         for ii in range(self.cutn):
            
            
#           # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
#           randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
#           size_mult = randsize ** self.cut_pow
#           size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
#           # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

#           offsetx = torch.randint(0, sideX - size + 1, ())
#           offsety = torch.randint(0, sideY - size + 1, ())
#           cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
#           cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        
        
#         cutouts = torch.cat(cutouts, dim=0)
#         cutouts = clamp_with_grad(cutouts, 0, 1)

#         #if args.use_augs:
#         cutouts = self.augs(cutouts)
#         if self.noise_fac:
#           facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
#           cutouts = cutouts + facs * torch.randn_like(cutouts)
#         return cutouts

# class MakeCutoutsCumin(nn.Module):
#     #from https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ
#     def __init__(self, cut_size, cutn, cut_pow, augs):
#         super().__init__()
#         self.cut_size = cut_size
#         print(f'cut size: {self.cut_size}')
#         self.cutn = cutn
#         self.cut_pow = cut_pow
#         self.noise_fac = 0.1
#         self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
#         self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
#         self.augs = augs
        
#         nn.Sequential(
#           #K.RandomHorizontalFlip(p=0.5),
#           #K.RandomSharpness(0.3,p=0.4),
#           #K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
#           #K.RandomGaussianNoise(p=0.5),
#           #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
#           K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
#           K.RandomPerspective(0.7,p=0.7),
#           K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
#           K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),)
            
#     def set_cut_pow(self, cut_pow):
#       self.cut_pow = cut_pow
    
#     def forward(self, input):
#         sideY, sideX = input.shape[2:4]
#         max_size = min(sideX, sideY)
#         min_size = min(sideX, sideY, self.cut_size)
#         cutouts = []
#         cutouts_full = []
#         noise_fac = 0.1
        
        
#         min_size_width = min(sideX, sideY)
#         lower_bound = float(self.cut_size/min_size_width)
        
#         for ii in range(self.cutn):
            
            
#           # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
#           randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
#           size_mult = randsize ** self.cut_pow
#           size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
#           # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

#           offsetx = torch.randint(0, sideX - size + 1, ())
#           offsety = torch.randint(0, sideY - size + 1, ())
#           cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
#           cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        
        
#         cutouts = torch.cat(cutouts, dim=0)
#         cutouts = clamp_with_grad(cutouts, 0, 1)

#         #if args.use_augs:
#         cutouts = self.augs(cutouts)
#         if self.noise_fac:
#           facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
#           cutouts = cutouts + facs * torch.randn_like(cutouts)
#         return cutouts


# class MakeCutoutsHolywater(nn.Module):
#     def __init__(self, cut_size, cutn, cut_pow, augs):
#         super().__init__()
#         self.cut_size = cut_size
#         print(f'cut size: {self.cut_size}')
#         self.cutn = cutn
#         self.cut_pow = cut_pow
#         self.noise_fac = 0.1
#         self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
#         self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
#         self.augs = augs
        
#         nn.Sequential(
#           #K.RandomHorizontalFlip(p=0.5),
#           #K.RandomSharpness(0.3,p=0.4),
#           #K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
#           #K.RandomGaussianNoise(p=0.5),
#           #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
#           K.RandomAffine(degrees=180, translate=0.5, p=0.2, padding_mode='border'),
#           K.RandomPerspective(0.6,p=0.9),
#           K.ColorJitter(hue=0.03, saturation=0.01, p=0.1),
#           K.RandomErasing((.1, .7), (.3, 1/.4), same_on_batch=True, p=0.2),)

#     def set_cut_pow(self, cut_pow):
#       self.cut_pow = cut_pow
    
#     def forward(self, input):
#         sideY, sideX = input.shape[2:4]
#         max_size = min(sideX, sideY)
#         min_size = min(sideX, sideY, self.cut_size)
#         cutouts = []
#         cutouts_full = []
#         noise_fac = 0.1
        
        
#         min_size_width = min(sideX, sideY)
#         lower_bound = float(self.cut_size/min_size_width)
        
#         for ii in range(self.cutn):
            
            
#           # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
#           randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
#           size_mult = randsize ** self.cut_pow
#           size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
#           # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

#           offsetx = torch.randint(0, sideX - size + 1, ())
#           offsety = torch.randint(0, sideY - size + 1, ())
#           cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
#           cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        
        
#         cutouts = torch.cat(cutouts, dim=0)
#         cutouts = clamp_with_grad(cutouts, 0, 1)

#         #if args.use_augs:
#         cutouts = self.augs(cutouts)
#         if self.noise_fac:
#           facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
#           cutouts = cutouts + facs * torch.randn_like(cutouts)
#         return cutouts


# class MakeCutoutsGinger(nn.Module):
#     def __init__(self, cut_size, cutn, cut_pow, augs):
#         super().__init__()
#         self.cut_size = cut_size
#         print(f'cut size: {self.cut_size}')
#         self.cutn = cutn
#         self.cut_pow = cut_pow
#         self.noise_fac = 0.1
#         self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
#         self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
#         self.augs = augs
#         '''
#         nn.Sequential(
#           K.RandomHorizontalFlip(p=0.5),
#           K.RandomSharpness(0.3,p=0.4),
#           K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
#           K.RandomGaussianNoise(p=0.5),
#           K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
#           K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
#           K.RandomPerspective(0.2,p=0.4, ),
#           K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),)
# '''

#     def set_cut_pow(self, cut_pow):
#       self.cut_pow = cut_pow

#     def forward(self, input):
#         sideY, sideX = input.shape[2:4]
#         max_size = min(sideX, sideY)
#         min_size = min(sideX, sideY, self.cut_size)
#         cutouts = []
#         cutouts_full = []
#         noise_fac = 0.1
        
        
#         min_size_width = min(sideX, sideY)
#         lower_bound = float(self.cut_size/min_size_width)
        
#         for ii in range(self.cutn):
            
            
#           # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
#           randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
#           size_mult = randsize ** self.cut_pow
#           size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
#           # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

#           offsetx = torch.randint(0, sideX - size + 1, ())
#           offsety = torch.randint(0, sideY - size + 1, ())
#           cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
#           cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        
        
#         cutouts = torch.cat(cutouts, dim=0)
#         cutouts = clamp_with_grad(cutouts, 0, 1)

#         #if args.use_augs:
#         cutouts = self.augs(cutouts)
#         if self.noise_fac:
#           facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
#           cutouts = cutouts + facs * torch.randn_like(cutouts)
#         return cutouts
'''
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
            K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomGaussianNoise(p=0.5),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
'''
# def load_vqgan_model(config_path, checkpoint_path):
#     config = OmegaConf.load(config_path)
#     if config.model.target == 'taming.models.vqgan.VQModel':
#         model = vqgan.VQModel(**config.model.params)
#         model.eval().requires_grad_(False)
#         model.init_from_ckpt(checkpoint_path)
#     elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
#         parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
#         parent_model.eval().requires_grad_(False)
#         parent_model.init_from_ckpt(checkpoint_path)
#         model = parent_model.first_stage_model
#     else:
#         raise ValueError(f'unknown model type: {config.model.target}')
#     del model.loss
#     return model

# def resize_image(image, out_size):
#     ratio = image.size[0] / image.size[1]
#     area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
#     size = round((area * ratio)**0.5), round((area / ratio)**0.5)
#     return image.resize(size, Image.LANCZOS)

# BUF_SIZE = 65536
# def get_digest(path, alg=hashlib.sha256):
#   hash = alg()
#   print(path)
#   with open(path, 'rb') as fp:
#     while True:
#       data = fp.read(BUF_SIZE)
#       if not data: break
#       hash.update(data)
#   return b64encode(hash.digest()).decode('utf-8')

# flavordict = {
#     "cumin": MakeCutoutsCumin,
#     "holywater": MakeCutoutsHolywater,
#     "ginger": MakeCutoutsGinger,
#     "custom": MakeCutoutsCustom
# }

class ModelHost:
  def __init__(self, args):
    self.args = args
    self.model, self.perceptor = None, None
    self.make_cutouts = None
    self.alt_make_cutouts = None
    self.imageSize = None
    self.prompts = None
    self.opt = None
    self.normalize = None
    self.z, self.z_orig, self.z_min, self.z_max = None, None, None, None
    self.metadata = None
    self.mse_weight = 0
    self.usealtprompts = False

  def setup_metadata(self, seed):
    metadata = {k:v for k,v in vars(self.args).items()}
    del metadata['max_iterations']
    del metadata['display_freq']
    metadata['seed'] = seed
    if (metadata['init_image']):
      path = metadata['init_image']
      digest = get_digest(path)
      metadata['init_image'] = (path, digest)
    if (metadata['image_prompts']):
      prompts = []
      for prompt in metadata['image_prompts']:
        path = prompt
        digest = get_digest(path)
        prompts.append((path,digest))
      metadata['image_prompts'] = prompts
    self.metadata = metadata

  def setup_model(self):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if self.args.prompts:
        print('Using prompts:', self.args.prompts)
    if self.args.altprompts:
        print('Using alternate augment set prompts:', self.args.altprompts)
    if self.args.image_prompts:
        print('Using image prompts:', self.args.image_prompts)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(f'{args.vqgan_model}.yaml', f'{args.vqgan_model}.ckpt').to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)

    make_cutouts = flavordict[flavor](cut_size, args.mse_cutn, cut_pow=args.mse_cut_pow,augs=args.augs)

    #make_cutouts = MakeCutouts(cut_size, args.mse_cutn, cut_pow=args.mse_cut_pow,augs=args.augs)
    if args.altprompts:
        self.usealtprompts = True
        self.alt_make_cutouts = flavordict[flavor](cut_size, args.mse_cutn, cut_pow=args.alt_mse_cut_pow,augs=args.altaugs)
        #self.alt_make_cutouts = MakeCutouts(cut_size, args.mse_cutn, cut_pow=args.alt_mse_cut_pow,augs=args.altaugs)
    
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    
    from PIL import Image
    pad_white = CC(sideX+ sideX//8)

    if args.init_image:
        pil_image = Image.open(args.init_image).convert('RGB')

        # enlarge image to fit in sideX, sideY, retaining its ratio
        if pil_image.size[0] > pil_image.size[1]:
            new_size = (sideX, int(sideX * pil_image.size[1] / pil_image.size[0]))
        else:
            new_size = (int(sideY * pil_image.size[0] / pil_image.size[1]), sideY)

        pil_image = pil_image.resize(new_size, Image.LANCZOS)
       
        # resize image using reflection padding
        # init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
        # m = nn.ReflectionPad2d(((sideX-pil_image.size[0])//2, (sideX-pil_image.size[0])//2, 
        #                        (sideY-pil_image.size[1])//2, (sideY-pil_image.size[1])//2))
        # init_img = m(init_img)
        # z, *_ = model.encode(init_img)
      
        # # resize image using white padding
        # # pad enlarged image to fit in sideX, sideY. Original image centered
        new_image = Image.new('RGB', (sideX, sideY), (255, 255, 255))
        new_image.paste(pil_image, ((sideX - new_size[0]) // 2, (sideY - new_size[1]) // 2))
        pil_image = new_image
        print("Init image size:", pil_image.size)

        init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
        z, *_ = model.encode(init_img)


        # pil_image = pad_white(pil_image)
        # pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        # init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
        # z, *_ = model.encode(init_img)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z = EMATensor(z, args.ema_val)
    
        # pil_image = Image.open(args.init_image).convert('RGB')
        # pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        # z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        # z = EMATensor(z, args.ema_val)

    if args.mse_with_zeros and not args.init_image:
        z_orig = torch.zeros_like(z.tensor)
    else:
        z_orig = z.tensor.clone()
        z_init = z.tensor.clone()

    z.requires_grad_(True)
    opt = optim.Adam(z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
    # opt = AdaBelief(z.parameters(), lr=args.mse_step_size, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

    self.cur_step_size =args.mse_step_size

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []
    altpMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))
    
    for prompt in args.altprompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        altpMs.append(Prompt(embed, weight, stop).to(device))
    
    from PIL import Image
    self.z_init = z_init
    self.init_img = init_img
    self.device = device
    self.cut_size = cut_size
    self.model, self.perceptor = model, perceptor
    self.make_cutouts = make_cutouts
    self.f = f
    self.imageSize = (sideX, sideY)
    self.prompts = pMs
    self.altprompts = altpMs
    self.opt = opt
    self.normalize = normalize
    self.z, self.z_orig, self.z_min, self.z_max = z, z_orig, z_min, z_max
    self.setup_metadata(seed)
    self.mse_weight = self.args.init_weight

    self.counter = 0
    # img_embeddings = self.embed_images(args.image_prompts)

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)

        # embed = self.embed_images([path])
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()

        # print("embed shape before: ", embed.shape)
        embed = embed - ovl_mean
        # print("embed shape after: ", embed.shape)

        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))
        if(self.usealtprompts):
          altpMs.append(Prompt(embed, weight).to(device))


  def embed_images_full(self, image_prompts, width=400, height=400):
      toksX, toksY = 400 // self.f, 400 // self.f
      sideX, sideY = toksX * self.f, toksY * self.f
      embeddings = []
      for prompt in image_prompts:
          path, weight, stop = parse_prompt(prompt)
          # print("path, weight, stop", path, weight, stop)
          img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
          img = resample(TF.to_tensor(img).unsqueeze(0).to(self.device), (self.cut_size, self.cut_size))
          embed = self.perceptor.encode_image(self.normalize(img)).float()
          embeddings.append(embed)

          # batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
          # embed = self.perceptor.encode_image(self.normalize(batch)).float()
      
      embeddings = clamp_with_grad(embeddings, 0, 1)
      embeddings = torch.cat(embeddings, dim=0)  

      return(embeddings)

  def embed_images_cuts(self, image_prompts, width=400, height=400):
      toksX, toksY = 400 // self.f, 400 // self.f
      sideX, sideY = toksX * self.f, toksY * self.f
      embeddings = []
      for path in image_prompts:
          # path, weight, stop = parse_prompt(prompt)
          # print("path, weight, stop", path, weight, stop)
          img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
          batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
          embed = self.perceptor.encode_image(self.normalize(batch)).float()
          embeddings.append(embed)
    
      embeddings = torch.cat(embeddings, dim=0)  

      return(embeddings)

  def synth(self, z):
      z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
      return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

  def add_metadata(self, path, i):
    imfile = PngImageFile(path)
    meta = PngInfo()
    step_meta = {'iterations':i}
    step_meta.update(self.metadata)
    #meta.add_itxt('vqgan-params', json.dumps(step_meta), zip=True)
    imfile.save(path, pnginfo=meta)

  @torch.no_grad()
  def checkin(self, i, losses, x):
      losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
      if i < args.mse_end:
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
      else:
        print(f'i: {i-args.mse_end} ({i}), loss: {sum(losses).item():g}, losses: {losses_str}')
      print(f'cutn: {self.make_cutouts.cutn}, cut_pow: {self.make_cutouts.cut_pow}, step_size: {self.cur_step_size}')
      out = self.synth(self.z.average)
      if i == self.args.max_iterations:
          if save_to_drive== True:
              batchpath = self.unique_index("./drive/MyDrive/VQGAN_Output/"+folder_name)
          else:
              batchpath = self.unique_index("./"+folder_name)
          TF.to_pil_image(out[0].cpu()).save(batchpath)
      #TF.to_pil_image(out[0].cpu()).save('progress.png')
      #self.add_metadata('progress.png', i)
      #display.display(display.Image('progress.png'))
      if self.args.png==True:
        TF.to_pil_image(out[0].cpu()).save('png_progress.png')
        self.add_metadata('png_progress.png', i)
        #I know it's a mess, BUT, it works, right? RIGHT?!
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
        import PIL.ImageOps    

        castle = Image.open(args.init_image).convert('RGB')
        #castle = Image.open('castle.png')
        castle = ImageEnhance.Brightness(castle)
        castle.enhance(1000).save('/content/png_processing/brightness.png')

        im = Image.open('/content/png_processing/brightness.png')
        im_invert = ImageOps.invert(im)
        im_invert.save('/content/png_processing/work.png')

        image = Image.open('/content/png_processing/work.png').convert('RGB')
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image.save('/content/png_processing/last.png')

        im_rgb = Image.open('progress.png')
        im_a = Image.open('/content/png_processing/last.png').convert('L').resize(im_rgb.size)
        im_rgb.putalpha(im_a)

        #im_rgb.save('/content/png_progress.png')
        im_rgb.save('/content/png_processing/progress.png')
        #display(Image.open('/content/png_progress.png').convert('RGB'))
        display(Image.open('/content/png_processing/progress.png'))

      else:
        TF.to_pil_image(out[0].cpu()).save('progress.png')
        self.add_metadata('progress.png', i)
        from PIL import Image
        display(Image.open('progress.png'))

  def unique_index(self, batchpath):
      i = 0
      while i < 10000:
          if os.path.isfile(batchpath+"/"+str(i)+".png"):
              i = i+1
          else:
              return batchpath+"/"+str(i)+".png"

  def ascend_txt(self, i):
      out = self.synth(self.z.tensor)
      iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()
      

      result = []
      if self.args.init_weight and self.mse_weight > 0:
          result.append(F.mse_loss(self.z.tensor, self.z_orig) * self.mse_weight / 2)
      
      ### INIT LOSS #########################
      # result.append(F.mse_loss(self.z.tensor, self.z_init) * 0.1)
      
      # mask = (-mh.init_img + 1).int()
      # result.append(F.mse_loss(out*mask, self.init_img*mask) * 0.5)

      _, out_edges = kornia.filters.canny(out)
      _, init_img_edges = kornia.filters.canny(self.init_img)
      result.append(F.mse_loss(out_edges[init_img_edges>0], init_img_edges[init_img_edges>0]) * 5)
      
      if not self.counter % 10:
        import matplotlib.pyplot as plt
        # convert back to numpy
        img_magnitude: np.ndarray = kornia.tensor_to_image(init_img_edges.byte())
        img_canny: np.ndarray = kornia.tensor_to_image(out_edges.byte())

        # Create the plot
        fig, axs = plt.subplots(1, 2, figsize=(16,16))
        axs = axs.ravel()

        axs[0].axis('off')
        axs[0].set_title('init image edge')
        axs[0].imshow(img_magnitude)

        axs[1].axis('off')
        axs[1].set_title('out init image edge')
        axs[1].imshow(img_canny)

        plt.show()

      self.counter += 1

      for prompt in self.prompts:
          result.append(prompt(iii))
          
      if self.usealtprompts:
        iii = self.perceptor.encode_image(self.normalize(self.alt_make_cutouts(out))).float()
        for prompt in self.altprompts:
          result.append(prompt(iii))
      
      img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
      img = np.transpose(img, (1, 2, 0))
      im_path = f'./steps/{i}.png'
      imageio.imwrite(im_path, np.array(img))
      self.add_metadata(im_path, i)
      return result

  def train(self, i,x):
      self.opt.zero_grad()
      mse_decay = self.args.mse_decay
      mse_decay_rate = self.args.mse_decay_rate
      lossAll = self.ascend_txt(i)

      if i < args.mse_end and i % args.mse_display_freq == 0:
        self.checkin(i, lossAll, x)
      if i == args.mse_end:
        self.checkin(i,lossAll,x)
      if i > args.mse_end and (i-args.mse_end) % args.display_freq == 0:
        self.checkin(i, lossAll, x)
         
      loss = sum(lossAll)
      loss.backward()
      self.opt.step()
      with torch.no_grad():
          if self.mse_weight > 0 and self.args.init_weight and i > 0 and i%mse_decay_rate == 0:
              self.z_orig = vector_quantize(self.z.average.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
              if self.mse_weight - mse_decay > 0:
                  self.mse_weight = self.mse_weight - mse_decay
                  print(f"updated mse weight: {self.mse_weight}")
              else:
                  self.mse_weight = 0
                  self.make_cutouts = flavordict[flavor](self.perceptor.visual.input_resolution, args.cutn, cut_pow=args.cut_pow, augs = args.augs)
                  if self.usealtprompts:
                      self.alt_make_cutouts = flavordict[flavor](self.perceptor.visual.input_resolution, args.cutn, cut_pow=args.alt_cut_pow, augs = args.altaugs)
                  self.z = EMATensor(self.z.average, args.ema_val)
                  self.new_step_size =args.step_size
                  self.opt = optim.Adam(self.z.parameters(), lr=args.step_size, weight_decay=0.00000000)
                  print(f"updated mse weight: {self.mse_weight}")
          if i > args.mse_end:
              if args.step_size != args.final_step_size and args.max_iterations > 0:
                progress = (i-args.mse_end)/(args.max_iterations)
                self.cur_step_size = lerp(step_size, final_step_size,progress)
                for g in self.opt.param_groups:
                  g['lr'] = self.cur_step_size
          #self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

  def run(self,x):
    i = 0
    try:
        pbar = tqdm(range(int(args.max_iterations + args.mse_end)))
        while True:
          self.train(i,x)
          if i > 0 and i%args.mse_decay_rate==0 and self.mse_weight > 0:
            self.z = EMATensor(self.z.average, args.ema_val)
            self.opt = optim.Adam(self.z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
          if i >= args.max_iterations + args.mse_end:
            pbar.close()
            break
          self.z.update()
          i += 1
          pbar.update()
    except KeyboardInterrupt:
        pass
    return i


import os
import random
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