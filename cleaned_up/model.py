import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF
from adabelief_pytorch import AdaBelief
from CLIP import clip
import kornia
import kornia.augmentation as K

from PIL import Image
import imageio

import wandb

from utils import *

class ModelHost:
    def __init__(self, args):
        self.args = args
        self.setup_model()

    def setup_model(self, args=None):
        if args: self.args = args
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.augs = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomSharpness(0.3,p=0.4),
                K.RandomGaussianBlur((3,3),(4.5,4.5),p=0.3),
                #K.RandomGaussianNoise(p=0.5),
                #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
                K.RandomPerspective(0.2,p=0.4, ),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),).to(device)
        self.ovl_mean_sketch = torch.load("results/ovl_mean_sketch.pth")

        print('Using device:', device)
        if self.args.prompts:
            print('Using prompts:', self.args.prompts)
        if self.args.image_prompts:
            print('Using image prompts:', self.args.image_prompts)
        if self.args.seed is None:
            seed = torch.seed()
        else:
            seed = self.args.seed
        torch.manual_seed(seed)
        print('Using seed:', seed)

        model = load_vqgan_model(f'{self.args.vqgan_model}.yaml', f'{self.args.vqgan_model}.ckpt').to(device)
        perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
        # perceptor.encode_image = nn.DataParallel(perceptor.encode_image, device_ids=[0, 1, 2, 3])

        cut_size = perceptor.visual.input_resolution
        e_dim = model.quantize.e_dim
        f = 2**(model.decoder.num_resolutions - 1)
        make_cutouts = Cutouts_preset(cut_size, self.args.cutn, cut_pow=self.args.cut_pow,augs=self.augs)

        n_toks = model.quantize.n_e
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f

        if self.args.init_image:
            pil_image = Image.open(self.args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
            z, *_ = model.encode(init_img)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

        z = ParamTensor(z)
        z_init = z.tensor.clone()
        z.requires_grad_(True)

        if self.args.optim == 'adam':
            if self.args.weight_decouple:
                opt = optim.AdamW(z.parameters(), lr=self.args.step_size, 
                                weight_decay=self.args.weight_reg, betas=(self.args.beta1, 0.999),
                                eps = self.args.epsilon)         
            else:
                opt = optim.Adam(z.parameters(), lr=self.args.step_size, 
                                weight_decay=self.args.weight_reg, betas=(self.args.beta1, 0.999),
                                eps = self.args.epsilon)
        elif self.args.optim == 'adabelief':
            opt = AdaBelief(z.parameters(), lr=self.args.step_size, 
                            weight_decay=self.args.weight_reg, betas=(self.args.beta1, 0.999),
                            eps = self.args.epsilon, weight_decouple=self.args.weight_decouple,
                            rectify = self.args.rectify)

        self.normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        pMs = []
        for prompt in self.args.prompts:
            print(prompt)
            txt, weight, stop = parse_prompt(prompt)
            print(txt, weight, stop)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))
        
        if self.args.sketch_embed_weight:
            pMs.append(Prompt(self.ovl_mean_sketch, self.args.sketch_embed_weight).to(device))

        # Make sure this is added as the final prompt
        for prompt in self.args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = perceptor.encode_image(self.normalize(batch)).float()
            embed = embed - self.ovl_mean_sketch
            pMs.append(Prompt(embed, weight, stop).to(device))

        if self.args.init_image:
            self.init_img = init_img
        self.device = device
        self.cut_size = cut_size
        self.model, self.perceptor = model, perceptor
        self.make_cutouts = make_cutouts
        self.f = f
        self.imageSize = (sideX, sideY)
        self.prompts = pMs
        self.opt = opt
        self.z, self.z_init = z, z_init
    
    @torch.no_grad()
    def reset_img_prompt(self):
        batch = self.make_cutouts(self.latest_out.to(self.device))
        embed = self.perceptor.encode_image(self.normalize(batch)).float()
        embed = embed - self.ovl_mean_sketch
        self.prompts[-1] = (Prompt(embed, 1).to(self.device))

    def embed_images_full(self, image_prompts, width=400, height=400):
        toksX, toksY = 400 // self.f, 400 // self.f
        sideX, sideY = toksX * self.f, toksY * self.f
        embeddings = []
        for prompt in image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            embed = self.perceptor.encode_image(self.normalize(img)).float()
            embeddings.append(embed)
        embeddings = clamp_with_grad(embeddings, 0, 1)
        embeddings = torch.cat(embeddings, dim=0)  
        return(embeddings)

    def embed_images_cuts(self, image_prompts, width=400, height=400):
        toksX, toksY = width // self.f, width // self.f
        sideX, sideY = toksX * self.f, toksY * self.f
        embeddings = []
        from tqdm import tqdm
        for path in tqdm(image_prompts):
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            embeddings.append(embed)
        embeddings = torch.cat(embeddings, dim=0)  
        return(embeddings)

    def synth(self, z):
        z_q = vector_quantize(z.tensor.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')

    def unique_index(self, batchpath):
        i = 0
        while i < 10000:
            if os.path.isfile(batchpath+"/"+str(i)+".png"):
                i = i+1
            else:
                return batchpath+"/"+str(i)+".png"
                
    def ascend_txt(self, i):
        import time

        # now = time.time()
        out = self.synth(self.z)
        # print(f"synth took {time.time() - now}")

        # now = time.time()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()
        # print(f"perceptor took {time.time() - now}")
        
        result = []        
        if self.args.edge_weight:
            _, out_edges = kornia.filters.canny(out)
            _, init_img_edges = kornia.filters.canny(self.init_img)
            edge_loss = F.mse_loss(out_edges[init_img_edges>0], init_img_edges[init_img_edges>0])
            result.append(edge_loss * self.args.edge_weight)
        
        text_loss = dict()
        for j, prompt in enumerate(reversed(self.prompts)):
            if not j: concept_loss = prompt(iii)
            else: text_loss[j] = prompt(iii)
            result.append(prompt(iii))
        
        if not i % self.args.images_interval: 
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
            img = np.transpose(img, (1, 2, 0))
            im_path = f'outputs/{self.args.experiment_name}/{i}.png'
            imageio.imwrite(im_path, np.array(img))
        
            if self.args.wandb:
                wandb.log({'edge_loss': edge_loss})
                wandb.log({'concept_loss': concept_loss})
                for key, value in text_loss.items():
                    wandb.log({f'text_loss_{key}': value})
                wandb.log({f"step {i}": wandb.Image(im_path)})

                if self.args.log_edges != 0:
                    init_edges: np.ndarray = kornia.tensor_to_image(init_img_edges.byte())
                    out_edges: np.ndarray = kornia.tensor_to_image(out_edges.byte())
                    mask_img = wandb.Image(im_path, masks={
                            "out_edges": {
                                "mask_data": out_edges.astype(int),
                            },
                            "init_edges": {
                                "mask_data": init_edges.astype(int)
                            }
                        })
                    wandb.log({f"step {i} w/ edges": mask_img})

        self.latest_out = out
        return result

    def step(self, i):
        self.opt.zero_grad()
        lossAll = self.ascend_txt(i)
        if not i % self.args.images_interval: self.checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()

    def train(self):
        if self.args.wandb:
            os.makedirs(f"outputs/{self.args.experiment_name}", exist_ok=True)
        i = 0
        try:
            for i in range(self.args.max_iterations+1):
                self.step(i)
                if self.args.reset_img_prompt_every and self.args.reset_img_prompt_every % i:
                    self.reset_img_prompt()
        except KeyboardInterrupt:
            pass
