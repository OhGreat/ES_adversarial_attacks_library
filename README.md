# Understanding ES Adversarial Attacks
This repository contains a framework to prototype black-box adversarial attacks on DNNs, by using Evolutionary Strategies.
A Grad-CAM implementation following the original publication in <a href="#original_publication">[1]</a> has also been created to visualize the impact of the adversarial attacks with respect to the input.

The purpose of this work is twofold, aiming to quantify the robustness of different (classification) DNN architectures on Black-Box attacks, while also qualitatively understanding how the attacks affect the focus of the model on the input images.

## Implementations
A number of different evaluation functions has been created in order to model the adversarial attacks. More specifically, the following modes have been implemented:
- "R_channel_only" : creates noise for each pixel in the red channel.
- "all_channels" : creates noise for each pixel in every channel.
- "shadow_noise" : like the R_channel_only but applies the noise to all channels. 
- "1D_one-pixel" : one pixel noise applied to only one channel.
- "3D_one-pixel" : one pixel noise for each channel.

Models used:
- VGG19
- ResNet50
- XceptionV3
- ViT (to be implemented)


## Installation
To use the repository a Python 3 environment is required. It is recommended to use Anaconda as a virtual environment as the installation for pytorch becomes pretty straightforward. After creating an anaconda environment you can use the following commands to install the required packages:

To install pytorch:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
To install the remaining packages you can use the `requirements.txt` file in the main directory as below:
```bash
pip install -r requirements.txt
```

## Usage

## References
<div id="original_publication">
[1].<br/>
R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.</a>
</div>
