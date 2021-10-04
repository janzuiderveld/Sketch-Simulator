# Sketch Simulator
An architecture that makes any doodle realistic, in any specified style, using VQGAN, CLIP and some basic embedding arithmetics.

## Architecture overview
![](https://i.ibb.co/SJxKby4/image.png)

## Setup
* run `setup.sh` in your environment. This will install required libraries and download model weights.

## Usage
* To work a single doodle, in your desired style (see `train.py` for all avaible modifiers), run: 
  * `train.py --start_image "path/to/your/doodle" --prompts "a painting in the style of ... | Trending on artstation`   

* To explore the hyperparameter space or large amounts of doodles and / or promps using weights and biases:
  * Create a sweep config with your desired parameters 'your_sweep.yaml' (see `sweep_configs/*` for examples)
  * Start the sweep:
    * `wandb sweep -p Sketch-sim "\path\to\your_sweep.yaml"` (this returns the sweep_ID, to be used in the next command)
    * `wandb agent janzuiderveld/Sketch-sim/sweep_ID''`
  * Alternatively, when working in SLURM environments, one can utilize `SLURM_scripts/sweeper.sh' (make sure to edit paths appropriately):
    * `sbatch SLURM_scripts/sweeper.sh "path/to/your_sweep.yaml"`

* To (re)calculate average sketch embeddings (`results/ovl_mean_sketch.pth` is calculated based on 1000 (padded) items per class for all 350 quickdraw classes) run:
  * `extract_sketch_emb.py --items_per_class 1000 --save_root "path/to/repo/root" --pad_images 6`
