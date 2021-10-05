# Sketch Simulator
An architecture that makes any doodle realistic, in any specified style, using VQGAN, CLIP and some basic embedding arithmetics.

See the final cell output of the colab below for some examples with and without subtracting sketch embedding averages.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JP6ExVr91ACXYA-0cBO_YTpeKoQgXTrW?usp=sharing)

*WARNING: This colab is messy, a precursor of the code in this repo, but it works.*


## Architecture Overview
![](https://i.ibb.co/SJxKby4/image.png)

## Setup
* run `setup.sh` in your environment. This will install required libraries and download model weights.

## Usage
* To work a single doodle, in your desired style (see `train.py` for all avaible modifiers), run: 
  * `train.py --start_image "path/to/your/doodle" --prompts "a painting in the style of ... | Trending on artstation`   
  
  Prompts are split using "|", and specific weights can be assigned using `{prompt1}:{weight1}|{prompt2}:{weight2}`

* To explore the hyperparameter space or large amounts of doodles and / or promps using weights and biases:
  * Create a sweep config with your desired parameters `your_sweep.yaml` in `sweep_configs/` (see `sweep_configs/*` for examples)
  * Start the sweep:
    * `wandb sweep -p Sketch-sim "\path\to\your_sweep.yaml"` (this returns the sweep_ID, to be used in the next command)
    * `wandb agent janzuiderveld/Sketch-sim/sweep_ID''`
  * Alternatively, when working in SLURM environments, one can utilize `SLURM_scripts/sweeper.sh' (make sure to edit paths appropriately):
    * `sbatch SLURM_scripts/sweeper.sh "path/to/your_sweep.yaml"`

All outputs are saved in `outputs/{args.experiment_name}/step_{i}.png`


## Calculate Average Sketch Embedding
* To (re)calculate average sketch embeddings (`results/ovl_mean_sketch.pth` is calculated based on 1000 (padded) items per class for all 350 quickdraw classes) run:
  * `extract_sketch_emb.py --items_per_class 1000 --save_root "path/to/repo/root" --pad_images 6`

## Notes
* 1 step of synthesizing + embedding 400x400 images takes about 0.3 seconds on a single 1080, usually 20-30 steps is enough for nice results.
* Prompts can be used as a metric in large hyperparameter sweeps (their scores are automatically logged) by using a weight of 0.

## TODO
* Add server / client scripts to circumvent startup times
* Add CLIP-based classifier for testing conceptual embedding accuracy on Quickdraw classification

