import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF
from torchvision import transforms
from adabelief_pytorch import AdaBelief
from CLIP import clip
import kornia
import kornia.augmentation as K

from PIL.PngImagePlugin import PngImageFile, PngInfo
from PIL import Image
import imageio

import wandb

from utils import *

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
    self.setup_model()

  def setup_metadata(self, seed):
    metadata = {k:v for k,v in vars(self.args).items()}
    try:
        metadata = metadata['_items']
    except:
        pass
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
    # self.args.init_img = self.args.start_image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if self.args.prompts:
        print('Using prompts:', self.args.prompts)
    if self.args.altprompts:
        print('Using alternate augment set prompts:', self.args.altprompts)
    if self.args.start_image:
        print('Using image prompts:', self.args.start_image)
    seed = torch.seed()
    torch.manual_seed(seed)
    print('Using seed:', seed)

    print('Using prompt:', self.args.prompts)
    model = load_vqgan_model(f'{self.args.vqgan_model}.yaml', f'{self.args.vqgan_model}.ckpt').to(device)
    perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)

    augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomGaussianBlur((3,3),(4.5,4.5),p=0.3),
            #K.RandomGaussianNoise(p=0.5),
            #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
            K.RandomPerspective(0.2,p=0.4, ),
            
            # induces background details
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.35), # og p =0.7

            # This one really needed?
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.35), # og p =0.7 
            )

    augs_init = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomGaussianBlur((3,3),(4.5,4.5),p=0.3),

            #K.RandomGaussianNoise(p=0.5),
            #K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            # K.RandomAffine(degrees=60, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
            K.RandomAffine(degrees=60, translate=0, shear=5, p=0.8, padding_mode='border'), # padding_mode=2
            K.RandomPerspective(0.2,p=0.4, ),
            
            # K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
            )

    make_cutouts_init = flavordict[self.args.flavor](cut_size, self.args.init_cutn, cut_pow=self.args.init_cut_pow, augs=augs_init)
    make_cutouts = flavordict[self.args.flavor](cut_size, self.args.cutn, cut_pow=self.args.cut_pow,augs=augs)
    make_cutouts_det = flavordict["det"](cut_size, self.args.cutn, cut_pow=self.args.cut_pow,augs=augs)
    
    n_toks = model.quantize.n_e
    toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    
    from PIL import Image

    if self.args.init_image:
        init_img = self.load_init_image(self.args.init_image, sideX, sideY, device) 
        
        # pil_image = Image.open(self.args.init_image).convert('RGB')
        # pil_image = self.resize_image_custom(pil_image, sideX, sideY, padding=self.args.padding)
        # init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0)
        
        # pil_image = resize_image(pil_image, (sideX, sideY))
        # init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
        
        z, *_ = model.encode(init_img * 2 - 1)
        # z, *_ = model.encode(init_img)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z = EMATensor(z, self.args.ema_val)
    
        # pil_image = Image.open(self.args.init_image).convert('RGB')
        # pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        # z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        # z = EMATensor(z, self.args.ema_val)

    z_orig = z.tensor.clone()
    z_init = z.tensor.clone()

    z.requires_grad_(True)
    opt = optim.Adam(z.parameters(), lr=self.args.step_size, weight_decay=0.00000000)
    # opt = AdaBelief(z.parameters(), lr=self.args.step_size, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

    self.cur_step_size =self.args.step_size

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []
    altpMs = []

    for prompt in self.args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        if txt:
          embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float().unsqueeze(0)
          pMs.append(Prompt(embed, weight, stop, name="text").to(device))
    
        # txt_embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()

    for prompt in self.args.altprompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        altpMs.append(Prompt(embed, weight, stop).to(device))
    
    from PIL import Image
    # img_embeddings = self.embed_images(self.args.start_image)

    self.z_init = z_init
    self.init_img = init_img
    self.device = device
    self.cut_size = cut_size
    self.model, self.perceptor = model, perceptor
    self.make_cutouts_init = make_cutouts_init
    self.make_cutouts = make_cutouts
    self.make_cutouts_det = make_cutouts_det
    self.f = f
    self.imageSize = (sideX, sideY)
    self.prompts = pMs
    self.altprompts = altpMs
    self.opt = opt
    self.normalize = normalize
    self.z, self.z_orig, self.z_min, self.z_max = z, z_orig, z_min, z_max
    # self.setup_metadata(seed)
    self.mse_weight = self.args.init_weight

    self.counter = 0

    if self.args.embedding_tgt:
      tgt_embed = torch.load(self.args.embedding_tgt).reshape(1, 1, 512)
      pMs.append(Prompt(tgt_embed, 3, -np.inf, name="tgt_embed").to(device))

    # IMAGE CONTENT PROMPT BIZZ ##########################$########

    path, weight, stop = parse_prompt(self.args.start_image)
    print("image weight", weight)
    # img = self.resize_image_custom(Image.open(path).convert('RGB'), sideX, sideY)

    ovl_mean = torch.load(self.args.embedding_avg)

    if "0" in self.args.embedding_avg:
      ovl_mean = torch.zeros_like(ovl_mean)
      print("zero")

    # set random cuts as target. Prompt class uses alll cuts, todo check how the distance to all of these is calculated, avg?
    if self.args.target_avg_cuts:
        batch = make_cutouts_init(init_img)
        embed = perceptor.encode_image(normalize(batch.squeeze())).float().unsqueeze(0)
        
        embed = (embed - ovl_mean) 
    
        # embed = (embed - ovl_mean + txt_embed) 

        # Target ovl_mean as style for testing ############    
        # tgt_style = torch.load("/content/Sketch-Simulator/results/ovl_mean_sketchy_cutouts.pt")    
        # embed = (embed - ovl_mean + tgt_style) 

        # Target ovl_mean for testing ############        
        # embed = ovl_mean.unsqueeze(0)

        # Target a image for testing ############        
        # init_img = self.load_init_image("/content/Sketch-Simulator/test_images/0.png", sideX, sideY, device)
        # batch = make_cutouts_init(init_img)
        # embed = perceptor.encode_image(normalize(batch)).float()

        pMs.append(Prompt(embed, weight, stop, name="image").to(device))
        print("embed target", Prompt(embed, weight, stop).embed.shape)

    if self.args.target_full_img:
        embed = self.embed_images_full([self.args.init_image])
        embed = embed - ovl_mean
        pMs.append(Prompt(embed, weight, stop, name="image").to(device))
        print("embed full target", Prompt(embed, weight, stop).embed.shape)

    if self.args.target_det_cuts:
        # use TORCH.NN.FUNCTIONAL.GRID_SAMPLE and TORCH.NN.FUNCTIONAL.AFFINE_GRID?
        # torch.nn.functional.affine_grid()

        batch, levels = make_cutouts_det(init_img, init=True)
        # print(batch.shape)
        batch = make_cutouts_init(batch)
        # print(batch.shape)


############################
        embed =[]
        for i, cutout_set in enumerate(batch):
          embed.append(perceptor.encode_image(normalize(cutout_set)).float().unsqueeze(0))
        embed = torch.cat(embed, dim = 0)
########################
        # embed = perceptor.encode_image(normalize(batch)).float()

        print("embed.shape")
        print(embed.shape)
        ovl_mean = ovl_mean.repeat(1, 256, 1)
        print(embed.shape)
        
        # embed = embed - ovl_mean + txt_embed
        embed = embed - ovl_mean.repeat(1, 256, 1)


        pMs.append(Prompt(embed, weight, stop, name="image", levels=levels, levels_bool=True, cutn=self.args.cutn, init_cutn=self.args.init_cutn).to(device))


    # print("embed shape before: ", embed.shape)
    # print("embed shape after: ", embed.shape)


    for weight in self.args.noise_prompt_weights:
        gen = torch.Generator().manual_seed(-1)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))
        if(self.usealtprompts):
          altpMs.append(Prompt(embed, weight).to(device))

  def load_init_image(self, image_path, sideX, sideY, device):
    pil_image = Image.open(image_path).convert('RGB')
    pil_image = self.resize_image_custom(pil_image, sideX, sideY, padding=self.args.padding)
    init_img = TF.to_tensor(pil_image).to(device).unsqueeze(0)
    return init_img

  def resize_image_custom(self, pil_image, sideX, sideY, padding = 0):
    # enlarge image to fit in sideX, sideY, retaining its ratio
    if pil_image.size[0] > pil_image.size[1]:
        new_size = (sideX, int(sideX * pil_image.size[1] / pil_image.size[0]))
    else:
        new_size = (int(sideY * pil_image.size[0] / pil_image.size[1]), sideY)

    if padding > 0:
        new_size = (new_size[0] - padding, new_size[1] - padding)

    pil_image = pil_image.resize(new_size, Image.LANCZOS)
    print(pil_image.size)
    # # resize image using white padding
    # # pad enlarged image to fit in sideX, sideY. Original image centered
    new_image = Image.new('RGB', (sideX, sideY), (255, 255, 255))
    new_image.paste(pil_image, ((sideX - new_size[0]) // 2, (sideY - new_size[1]) // 2))
    pil_image = new_image
    print("Init image size:", pil_image.size)
    return pil_image

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
      
    #   embeddings = clamp_with_grad(embeddings, 0, 1)
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
    # step_meta.update(self.metadata)
    #meta.add_itxt('vqgan-params', json.dumps(step_meta), zip=True)
    imfile.save(path, pnginfo=meta)

  @torch.no_grad()
  def checkin(self, losses):
      losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
      print(f'i: {self.counter}, loss: {sum(losses).item():g}, losses: {losses_str}')
      print(f'cutn: {self.make_cutouts.cutn}, cut_pow: {self.make_cutouts.cut_pow}, step_size: {self.cur_step_size}')
      out = self.synth(self.z.average)
      if self.counter == self.args.max_iterations:
          os.makedirs(self.args.output_dir, exist_ok=True)
          batchpath = self.unique_index(self.args.output_dir)
        #   batchpath = self.args.output_dir + '/' + self.args.init_image.split('/')[-1] 

          input_img_array = TF.to_pil_image(self.init_img.cpu().squeeze())
          output_img_array = TF.to_pil_image(out[0].cpu())
        
          if self.args.save_bef_aft:
            # concatenate input and output images in a single image
            im = Image.new('RGB', (input_img_array.width + output_img_array.width, input_img_array.height))
            im.paste(input_img_array, (0, 0))
            im.paste(output_img_array, (input_img_array.width, 0))
            im.save(batchpath)
          else:
            output_img_array.save(batchpath)

          if self.args.wandb:
              # wandb.log({'edge_loss': edge_loss})
              # wandb.log({'concept_loss': concept_loss})
              # for key, value in text_loss.items():
              #     wandb.log({f'text_loss_{key}': value})
              
              wandb.log({
                        f"{self.args.init_image[0].split('/')[-1]} out": wandb.Image(batchpath),
                        f"{self.args.init_image[0].split('/')[-1]} init": wandb.Image(self.args.init_image)
                        })
              wandb.log({})

              # if self.args.log_edges != 0:
              #     init_edges: np.ndarray = kornia.tensor_to_image(init_img_edges.byte())
              #     out_edges: np.ndarray = kornia.tensor_to_image(out_edges.byte())
              #     mask_img = wandb.Image(im_path, masks={
              #             "out_edges": {
              #                 "mask_data": out_edges.astype(int),
              #             },
              #             "init_edges": {
              #                 "mask_data": init_edges.astype(int)
              #             }
              #         })
              #     wandb.log({f"step {i} w/ edges": mask_img})

      #TF.to_pil_image(out[0].cpu()).save('progress.png')
      #self.add_metadata('progress.png', i)
      #display.display(display.Image('progress.png'))
      

      TF.to_pil_image(out[0].cpu()).save('progress.png')
      self.add_metadata('progress.png', self.counter)

  def unique_index(self, batchpath):
      i = 0
      fname = self.args.init_image.split('/')[-1].split(".")[0].split(" ")[0] + "_" + self.args.embedding_avg.split("/")[-1].split(".")[0].replace("_", "") + "_" + self.args.prompts[0].replace(" ", "") 
      while True:
          if os.path.isfile(batchpath+"/"+ fname + "_" +str(i)+".png"):
              i = i+1
          else:
              return batchpath+"/"+ fname + "_" +str(i)+".png"

  def ascend_txt(self):
      out = self.synth(self.z.tensor)
      if self.args.target_det_cuts:
        out_grid = self.make_cutouts_det(out)
        # print(out_grid.shape)
        out_grid_cuts = self.make_cutouts(out_grid)
        # print(out_grid_cuts.shape)
      

        iii =[]
        for i, cutout_set in enumerate(out_grid_cuts):
          iii.append(self.perceptor.encode_image(self.normalize(cutout_set)).float().unsqueeze(0))
        iii = torch.cat(iii, dim = 0)
        
        # iii = self.perceptor.encode_image(self.normalize(out_grid_cuts)).float()


      else:


        # iii =[]
        # for i, cutout_set in enumerate(out_grid_cuts):
        #   iii.append(self.perceptor.encode_image(self.normalize(cutout_set)).float().unsqueeze(0))
        
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out).squeeze())).float().unsqueeze(0)



      result = []
      if self.args.init_weight and self.mse_weight > 0:
          result.append(F.mse_loss(self.z.tensor, self.z_orig) * self.mse_weight / 2)
      
      ### INIT LOSS #########################
      # result.append(F.mse_loss(self.z.tensor, self.z_init) * 0.1)
      
      # mask = (-mh.init_img + 1).int()
      # result.append(F.mse_loss(out*mask, self.init_img*mask) * 0.5)

      _, out_edges = kornia.filters.canny(out)
      _, init_img_edges = kornia.filters.canny(self.init_img)
      result.append(F.mse_loss(out_edges[init_img_edges>0], init_img_edges[init_img_edges>0]) * self.args.edge_weight)
      
      # if not self.counter % 10:
      #   import matplotlib.pyplot as plt
      #   # convert back to numpy
      #   img_magnitude: np.ndarray = kornia.tensor_to_image(init_img_edges.byte())
      #   img_canny: np.ndarray = kornia.tensor_to_image(out_edges.byte())

      #   # plot the results ##########################################
      #   # Create the plot
      #   fig, axs = plt.subplots(1, 2, figsize=(16,16))
      #   axs = axs.ravel()

      #   axs[0].axis('off')
      #   axs[0].set_title('init image edge')
      #   axs[0].imshow(img_magnitude)

      #   axs[1].axis('off')
      #   axs[1].set_title('out init image edge')
      #   axs[1].imshow(img_canny)

      #   plt.show()

      self.counter += 1

      for prompt in self.prompts:
          result.append(prompt(iii))
          
      if self.usealtprompts:
        iii = self.perceptor.encode_image(self.normalize(self.alt_make_cutouts(out))).float()
        for prompt in self.altprompts:
          result.append(prompt(iii))
      
      img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
      img = np.transpose(img, (1, 2, 0))
      im_path = f'./steps/{self.counter}.png'
      imageio.imwrite(im_path, np.array(img))
      self.add_metadata(im_path, self.counter)
      return result

  def train(self):
      self.opt.zero_grad()
    #   mse_decay = self.args.decay
      lossAll = self.ascend_txt()

      if self.counter % self.args.display_freq == 0 or self.counter == 1:
        self.checkin(lossAll)
         
      loss = sum(lossAll)
      loss.backward()
      self.opt.step()
      with torch.no_grad():
          if self.mse_weight > 0 and self.args.init_weight and self.counter > 0 and self.counter%self.args.decay_rate == 0:
              self.z_orig = vector_quantize(self.z.average.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
            #   self.mse_weight = self.mse_weight - mse_decay
            #   print(f"updated mse weight: {self.mse_weight}")

  def run(self):
    i = 0
    # try:
    while True:
        self.train()
        if self.counter > 0 and self.counter%self.args.decay_rate==0 and self.mse_weight > 0:
            self.z = EMATensor(self.z.average, self.args.ema_val)
            self.opt = optim.Adam(self.z.parameters(), lr=self.args.step_size, weight_decay=self.args.weight_decay)
        if self.counter >= self.args.max_iterations:
            break
        self.z.update()
    # except KeyboardInterrupt:
    #     pass
    return 
