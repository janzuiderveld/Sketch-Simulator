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

from utils_old import *

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
    if self.args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(f'{self.args.vqgan_model}.yaml', f'{self.args.vqgan_model}.ckpt').to(device)
    perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)

    make_cutouts = flavordict[flavor](cut_size, self.args.mse_cutn, cut_pow=self.args.mse_cut_pow,augs=self.args.augs)

    #make_cutouts = MakeCutouts(cut_size, self.args.mse_cutn, cut_pow=self.args.mse_cut_pow,augs=self.args.augs)
    if self.args.altprompts:
        self.usealtprompts = True
        self.alt_make_cutouts = flavordict[flavor](cut_size, self.args.mse_cutn, cut_pow=self.args.alt_mse_cut_pow,augs=self.args.altaugs)
        #self.alt_make_cutouts = MakeCutouts(cut_size, self.args.mse_cutn, cut_pow=self.args.alt_mse_cut_pow,augs=self.args.altaugs)
    
    n_toks = model.quantize.n_e
    toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    
    from PIL import Image
    pad_white = CC(sideX+ sideX//8)

    if self.args.init_image:
        pil_image = Image.open(self.args.init_image).convert('RGB')

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
    z = EMATensor(z, self.args.ema_val)
    
        # pil_image = Image.open(self.args.init_image).convert('RGB')
        # pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        # z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        # z = EMATensor(z, self.args.ema_val)

    if self.args.mse_with_zeros and not self.args.init_image:
        z_orig = torch.zeros_like(z.tensor)
    else:
        z_orig = z.tensor.clone()
        z_init = z.tensor.clone()

    z.requires_grad_(True)
    opt = optim.Adam(z.parameters(), lr=self.args.mse_step_size, weight_decay=0.00000000)
    # opt = AdaBelief(z.parameters(), lr=self.args.mse_step_size, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

    self.cur_step_size =self.args.mse_step_size

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []
    altpMs = []

    for prompt in self.args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))
    
    for prompt in self.args.altprompts:
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
    self.mse_weight = self.self.args.init_weight

    self.counter = 0
    # img_embeddings = self.embed_images(self.args.image_prompts)

    for prompt in self.args.image_prompts:
        path, weight, stop = parse_prompt(prompt)

        # embed = self.embed_images([path])
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()

        # print("embed shape before: ", embed.shape)
        embed = embed - ovl_mean
        # print("embed shape after: ", embed.shape)

        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
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
  def checkin(self, i, losses):
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
      self.add_metadata(im_path, self.counter)
      return result

  def train(self):
      self.opt.zero_grad()
      mse_decay = self.args.mse_decay
      mse_decay_rate = self.args.mse_decay_rate
      lossAll = self.ascend_txt()

      if i < args.mse_end and i % args.mse_display_freq == 0:
        self.checkin(lossAll)
      if i == args.mse_end:
        self.checkin(lossAll)
      if i > args.mse_end and (i-args.mse_end) % args.display_freq == 0:
        self.checkin(lossAll)
         
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

  def run(self):
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