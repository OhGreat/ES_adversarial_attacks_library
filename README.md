# ES Black-box adversarial attacks library
This repository contains a library to prototype and test black-box adversarial attacks on DNNs, by using Evolutionary Strategies.
A Grad-CAM implementation following the original publication in <a href="#gradcam">[1]</a> has also been created to visualize the impact of the adversarial attacks with respect to the input.

The purpose of this work is twofold, aiming to quantify the robustness of different (classification) DNN architectures on Black-Box attacks, while also qualitatively understanding how the attacks affect the focus of the model on the input images.


## Implementations
A number of different evaluation functions has been created in order to model the adversarial attacks. More specifically, the following modes have been implemented:
- "R_channel_only" : creates noise for each pixel in the red channel.
- "all_channels" : creates noise for each pixel in every channel.
- "shadow_noise" : like the R_channel_only but applies the noise to all channels. 
- "3D_one-pixel" : one pixel noise for each channel as <a href="#one_pixel_atk">[2]</a>.
- "1D_one-pixel" : like above but the noise is applied to only one channel.

The models implemented for the Grad-CAM are the following:
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
<div id="gradcam">
[1].<br/>
R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.</a>
</div>
<br/>
<div id="one_pixel_atk">
[2].<br/>
J. Su, D. V. Vargas and K. Sakurai, "One Pixel Attack for Fooling Deep Neural Networks," in IEEE Transactions on Evolutionary Computation, vol. 23, no. 5, pp. 828-841, Oct. 2019, doi: 10.1109/TEVC.2019.2890858.
</div>