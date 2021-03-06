import sys
import torch
import math
from torch import nn, optim
from torch.nn import functional as F
import hashlib
from base64 import b64encode
from omegaconf import OmegaConf
import kornia.augmentation as K
sys.path.append('./taming-transformers')
from taming.models import vqgan
from PIL import ImageFile, Image
import os 
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
try:
    import cv2
except:
    pass 

import random
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

def lerp(a, b, f):
    return (a * (1.0 - f)) + (b * f);

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf'), levels=None, levels_bool=False, cutn=0, init_cutn=0, name=""):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
        self.levels_bool = levels_bool
        self.levels = levels

        self.cutn = cutn
        self.init_cutn = init_cutn

        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # if levels:
        #     self.register_buffer('levels', levels)
        # else:
        self.name = name

    def forward(self, input):
        # input_normed = F.normalize(input, dim=1)
        # embed_normed = F.normalize(self.embed, dim=1)
        # print(input_normed.shape, embed_normed.shape)
        # if self.levels is not None:
        #     dists = []
        #     for i in range(input_normed.shape[0]):
        #         dist = input_normed[i:i+1, :, :].sub(embed_normed[:, i:i+1, :]).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        #         dists.append(dist)
        #     dists = torch.cat(dists, dim=0)
        #     # dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        #     dists = dists * self.weight.sign()
        # else:
        #     dists = self.cos(input_normed, embed_normed)
        #     dists = dists * self.weight.sign()
        # return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        
        
        # input_normed = F.normalize(input.unsqueeze(1), dim=2)
        # embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        input_normed = F.normalize(input.unsqueeze(2), dim=3)
        embed_normed = F.normalize(self.embed.unsqueeze(1), dim=3)

        print("lalalala", self.name, self.levels_bool)
        print(input_normed.shape, embed_normed.shape)
        # print(self.levels)
        if embed_normed.shape[0] != 1:
            dists = []
            
            
            # for i in range(len(self.levels)):
            #     dist = input_normed[i*self.cutn*len(self.levels):(i+1)*len(self.levels)*self.cutn, :, :].sub(embed_normed[i*self.init_cutn*len(self.levels):(i+1)*len(self.levels)*self.init_cutn:, :]).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            #     dists.append(dist)

            for i in range(len(self.levels)):
                dist = input_normed[i,:, :, :].sub(embed_normed[i,:, :, :]).norm(dim=2).div(2).arcsin().pow(2).mul(2)
                dists.append(dist)

            # for (level_input, level_embed) in zip(input_normed, embed_normed):
            #   print(level_input.shape, level_embed.shape)
            #   dist = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            #   dists.append(dist)

            
            dists = torch.cat(dists, dim=0)
            # dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            dists = dists * self.weight.sign()
       
        else:
            embed_normed = embed_normed.squeeze(1)
            input_normed = input_normed.reshape((input_normed.shape[0]*input_normed.shape[1], 1, -1))
            print(input_normed.shape, embed_normed.shape)
            dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            dists = dists * self.weight.sign()
       
        if 0:
            # step1 = input_normed.sub(embed_normed)
            # print(step1.shape)
            # dists = step1.norm(dim=2).div(2).arcsin().pow(2).mul(2)
            dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            dists = dists * self.weight.sign()
            # print(dists.shape)
            # print(self.name)

        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

class EMATensor(nn.Module):
    """implmeneted by Katherine Crowson"""
    def __init__(self, tensor, decay):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
        self.register_buffer('biased', torch.zeros_like(tensor))
        self.register_buffer('average', torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer('accum', torch.tensor(1.))
        self.update()
    
    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError('update() should only be called during training')

        self.accum *= self.decay
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    def forward(self):
        if self.training:
            return self.tensor
        return self.average

def save_tensor_as_img(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # tensor = tensor / 255.
    # tensor = tensor.clamp(0, 1)
    # tensor = tensor.to(torch.float32)
    # print(tensor.shape)
    pil_img = TF.to_pil_image(tensor[0].cpu())
    pil_img.save(save_path)

class MakeCutoutsDet(nn.Module):
    def __init__(self, cut_size, cutn=None, cut_pow=None, augs=None, cut_levels=2, testing=True):
        super().__init__()
        self.cut_size = cut_size
        print(f'cut size: {self.cut_size}')
        
        self.cut_levels = cut_levels
        self.cutn_levels = [i**i for i in range(1, self.cut_levels)]
        self.cutn = sum(self.cutn_levels) 

        self.cut_pow = "det"
        self.testing = testing   

        self.used_cutout_indices = []

    def forward(self, input, init=False):
        sideY, sideX = input.shape[2:4]
        cutouts = []

        levels = []

        # white pad input to be square
        if sideY > sideX:
            input = F.pad(input, (0, 0, 0, sideY - sideX), 'constant', 0)
        elif sideX > sideY:
            input = F.pad(input, (0, 0, sideX - sideY, 0), 'constant', 0)

        # if self.testing:
        #     # read torch array with cv2
        #     img_cv2 = cv2.imread('/content/Sketch-Simulator/test_images/eedb70bc-7a45-41cd-98e1-1f91f6285803.jpeg')
        #     # reshape to sideX x sideY
        #     img_cv2 = cv2.resize(img_cv2, (sideX, sideY))

        max_size = max(sideX, sideY)
        

        for level in range(1,self.cut_levels):
            coord = np.linspace(0, max_size, level+1, dtype=np.int)
            for i in range(len(coord)-1): 
                for j in range(len(coord)-1):
                    cutout = input[:, :, coord[i]:coord[i+1], coord[j]:coord[j+1]]
                     

                    if init:
                        # calculate average pixel value of cutout
                        cutout_avg = cutout.mean()
                        # if cutout_avg > 0.98 and level != 1: # if cutout is mostly white
                        if cutout_avg > 0.95 and level != 1: # if cutout is mostly white
                            continue
                    
                        self.used_cutout_indices.append((level, i, j))
                        cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
                        save_tensor_as_img(cutout, f'/content/Sketch-Simulator/thrash/cutout_{level}_{i}_{j}.png')
                        levels.append((level, i, j))
                        
                        # if self.testing:
                            # cv2.rectangle(img_cv2, (coord[j]+random.randint(-3, 3), coord[i]+random.randint(-3, 3)), (coord[j+1]+random.randint(-3, 3), coord[i+1]+random.randint(-3, 3)), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
                            # cv2.rectangle(img_cv2, (0,0, 800, 800), (0,0,0), 2)
                    else:
                        if (level, i, j) in self.used_cutout_indices:
                            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
                            levels.append((level, i, j))
                    
        
        # if self.testing and init:
        #     print(len(cutouts))
        #     os.makedirs("/content/Sketch-Simulator/thrash/", exist_ok=True)
        #     cv2.imwrite('/content/Sketch-Simulator/thrash/test_rectangles.jpg',img_cv2) 


        cutouts = torch.cat(cutouts, dim=0)
        levels = torch.tensor(levels)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        if init:
            print(self.used_cutout_indices)
            return cutouts, levels

        return cutouts

   
            

class MakeCutoutsCumin(nn.Module):
    #from https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ
    def __init__(self, cut_size, cutn, cut_pow, augs, init=False):
        super().__init__()
        self.cut_size = cut_size
        print(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = augs
        self.init = init
        
        # nn.Sequential(
        #   #K.RandomHorizontalFlip(p=0.5),
        #   #K.RandomSharpness(0.3,p=0.4),
        #   #K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
        #   #K.RandomGaussianNoise(p=0.5),
        #   #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
        #   K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
        #   K.RandomPerspective(0.7,p=0.7),
        #   K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
        #   K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
        #   )
            
    def set_cut_pow(self, cut_pow):
      self.cut_pow = cut_pow
    
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1
        
        
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)
        
        #############################
        overall_avg = torch.mean(input)
        # print(f'overall avg: {overall_avg}')
        ########################################
        
        for ii in range(self.cutn):

        ###########################
          avg_pixel = 99
        #   while avg_pixel > overall_avg*1.1: 
          thresh = overall_avg*2
          if self.init: 
            thresh = overall_avg*1.1
          while avg_pixel > overall_avg*2: 
        ###########################

            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
            size_mult = randsize ** self.cut_pow
            size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
            # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            #######################
            avg_pixel = torch.mean(cutout)
            ###########################


          ###################
          if cutout.shape[0] > 1:
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)).unsqueeze(1))
          ##############################
          else:
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))


        ###################
        if cutouts[0].shape[0] > 1:
          cutouts = torch.cat(cutouts, dim=1)
        ###############
        else:
          cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)


        ##################
        cutouts_list = []
        if len(cutouts.shape) == 5:
          for i, cutout_level in enumerate(cutouts):
            cutouts_list.append(self.augs(cutout_level).unsqueeze(0))
          cutouts = torch.cat(cutouts_list, dim = 0)
        ###############
        else:
          cutouts = self.augs(cutouts)

        # print(cutouts.shape)
        # print(cutouts[i].shape)
        
        
        if self.noise_fac:
          ###################        
          if len(cutouts.shape) == 5:
            facs = cutouts.new_empty([cutouts.shape[0], cutouts.shape[1], 1, 1, 1]).uniform_(0, self.noise_fac)
          ########################          
          else:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
          cutouts = cutouts + facs * torch.randn_like(cutouts)

          ###################        
          if len(cutouts.shape) != 5:
            cutouts = cutouts.unsqueeze(0)
          ########################          

        return cutouts


class MakeCutoutsHolywater(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        print(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = augs
        
        nn.Sequential(
          #K.RandomHorizontalFlip(p=0.5),
          #K.RandomSharpness(0.3,p=0.4),
          #K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
          #K.RandomGaussianNoise(p=0.5),
          #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
          K.RandomAffine(degrees=180, translate=0.5, p=0.2, padding_mode='border'),
          K.RandomPerspective(0.6,p=0.9),
          K.ColorJitter(hue=0.03, saturation=0.01, p=0.1),
          K.RandomErasing((.1, .7), (.3, 1/.4), same_on_batch=True, p=0.2),)

    def set_cut_pow(self, cut_pow):
      self.cut_pow = cut_pow
    
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1
        
        
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)
        
        for ii in range(self.cutn):
            
            
          # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
          randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
          size_mult = randsize ** self.cut_pow
          size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
          # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

          offsetx = torch.randint(0, sideX - size + 1, ())
          offsety = torch.randint(0, sideY - size + 1, ())
          cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
          cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        
        
        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        #if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
          facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
          cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsGinger(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        print(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = augs
        '''
        nn.Sequential(
          K.RandomHorizontalFlip(p=0.5),
          K.RandomSharpness(0.3,p=0.4),
          K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
          K.RandomGaussianNoise(p=0.5),
          K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
          K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
          K.RandomPerspective(0.2,p=0.4, ),
          K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),)
'''

    def set_cut_pow(self, cut_pow):
      self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1
        
        
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)
        
        for ii in range(self.cutn):
            
            
          # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
          randsize = torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound,1.)
          size_mult = randsize ** self.cut_pow
          size = int(min_size_width * (size_mult.clip(lower_bound, 1.))) # replace .5 with a result for 224 the default large size is .95
          # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

          offsetx = torch.randint(0, sideX - size + 1, ())
          offsety = torch.randint(0, sideY - size + 1, ())
          cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
          cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        
        
        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        #if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
          facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
          cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

BUF_SIZE = 65536
def get_digest(path, alg=hashlib.sha256):
  hash = alg()
  print(path)
  with open(path, 'rb') as fp:
    while True:
      data = fp.read(BUF_SIZE)
      if not data: break
      hash.update(data)
  return b64encode(hash.digest()).decode('utf-8')

flavordict = {
    "cumin": MakeCutoutsCumin,
    "holywater": MakeCutoutsHolywater,
    "ginger": MakeCutoutsGinger,
    "det": MakeCutoutsDet,
}
