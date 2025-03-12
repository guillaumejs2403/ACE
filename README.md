 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ACE:

This is the codebase for the CVPR 2023 paper [Adversarial Counterfactual Visual Explanations](http://arxiv.org/abs/2303.09962).

## Environment

Through anaconda, install our environment:

```bash
conda env create -f environment.yaml
conda activate ace
``` 

## Downloading pre-trained models

To use ACE, you must download the pretrained DDPM models. Please extract them to a folder of choice `/path/to/models`. We provide links and instructions to download all models.

Download Link:

* CelebA Models
    - Classifier and diffusion model: [Link](https://huggingface.co/guillaumejs2403/DiME)
* CelebaA HQ
    - Diffusion Model: [Link](https://huggingface.co/guillaumejs2403/ACE)
    - Classifier: Please download `checkpoints_decision_densenet.tar.gz `. The classifier is the `celebamaskhq/checkpoint.tar`. [Link](https://github.com/valeoai/STEEX/releases)
* BDDOIA/100k
    - Diffusion Model: :warning: This DDPM can only generate images from a warm-up stage. [Link](https://huggingface.co/guillaumejs2403/ACE)
    - Classifier: Please download `checkpoints_decision_densenet.tar.gz `. The classifier is the `bdd/checkpoint.tar`. [Link](https://github.com/valeoai/STEEX/releases)
* ImageNet: 
    - Diffusion Model: download the `256x256 diffusion (not class conditional)` DDPM model through the `openai/guided-diffusion` repo. [Link](https://github.com/openai/guided-diffusion).
    - Classifier: we used the pretrained models given by PyTorch.
* Evaluation Models
    - CelebA Oracle: [Link](https://huggingface.co/guillaumejs2403/DiME)
    - CelebA HQ Oracle: Please download `checkpoints_oracle_attribute.tar.gz`. The classifier is the `celebamaskhq/checkpoint.tar`. [Link](https://github.com/valeoai/STEEX/releases)
    - VGGFace2 Model: Please download the `resnet50_ft` model. [Link](https://github.com/cydonia999/VGGFace2-pytorch).
    - SiamSiam Model: Please download the ResNet-50 model trained with a *batch size of 256*. [Link](https://github.com/facebookresearch/simsiam#models-and-logs)


## Generating Adversarial Counterfactual Explanations

To generate counterfactual explanations, use the `main.py` python script. We added a commentary to every possible flag for you to know what they do. Nevertheless, in the `script` folder, we provided several scripts to generate all counterfactual explanations using our proposed method: ACE. 

We follow the same folder ordering as DiME. Please see all details in [DiME's repository](https://github.com/guillaumejs2403/DiME#extracting-counterfactual-explanations). Similarly, we took advantage of their multi-chunk processing -- more info in DiME's repo.

To reduce the GPU burden, we implemented a checkpoint strategy to enable counterfactual production on a reduced GPU setup. `--attack_joint_checkpoint True` sets this modality on. Please check this [repo](https://github.com/cybertronai/gradient-checkpointing#how-it-works) for a nice explanation and visualization. The flag `--attack_checkpoint_backward_steps n` uses `n` DDPM iterations before computing the backward gradients. **It is 100% recommended to use a higher `--attack_checkpoint_backward_steps` value and a batch size of 1 than `--attack_checkpoint_backward_steps 1` and a larger batch size!!!** 

When you finished processing all counterfactual explanations, we store the counterfactual and the pre-explanation. You can easily re-process the pre-explanations using the `postprocessing.py` python script.

## Evaluating ACE

We provided a generic code base to evaluate counterfactual explanation methods. All evaluation script filenames begin with `compute`. Please look at their arguments on each individual script.
Notes: 
* All evaluations are based on the file organization created with our file system.
* `compute_FID` and `compute_sFID` are bash scripts. The first argument is the `output_path` as in the main script. The second one is the experiment name. We implemented a third one, a temporal folder where everything will be computed - useful for testing multiple models at the same time.

## Citation

Is you found our code useful, please cite our work:
```
@inproceedings{Jeanneret_2023_CVPR,
    author    = {Jeanneret, Guillaume and Simon, Lo\"ic and Fr\'ed\'eric Jurie},
    title     = {Adversarial Counterfactual Visual Explanations},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023}
}
``` 


## Code Base

We based our repository on our previous work [Diffusion Models for Counterfactual Explanations](https://github.com/guillaumejs2403/DiME).
