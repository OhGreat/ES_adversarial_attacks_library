# ES Black-box adversarial attacks library
This repository contains a library to prototype and test black-box adversarial attacks on DNNs, by using Evolutionary Strategies.
A Grad-CAM implementation following the original publication in <a href="#gradcam">[1]</a> has also been created to visualize the impact of the adversarial attacks with respect to the input.

The purpose of this work is twofold, aiming to quantify the robustness of different (classification) DNN architectures on Black-Box attacks, while also qualitatively understanding how the attacks affect the focus of the model on the input images.


## Implementations
A number of different evaluation functions has been created in order to model the `adversarial attacks`. More specifically, the following modes have been implemented:
- "R_channel" : creates noise for each pixel in the red channel.
- "all_channels" : creates noise for each pixel in every channel. Similarly to what is presented in <a href="tiling_es">[3]</a>
- "shadow_noise" : like the *R_channel* but applies the noise to all three channels. 
- "3D_one-pixel" : one pixel noise for each channel as described in <a href="#one_pixel_atk">[2]</a>.
- "1D_one-pixel" : like above but the noise is applied to only one channel.

The implemented `models` for Grad-CAM visualization are the following:
- VGG19
- ResNet50
- XceptionV3
- VisionTransformer
- SwinTransformer


## Installation
To use the repository a `Python 3` environment is required. Using Anaconda is recommended. After creating an anaconda environment you can use the following commands to install the required packages:

To install pytorch:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
To install the remaining packages you can use the `requirements.txt` file in the main directory as below:
```bash
pip install -r requirements.txt
```


## Usage

### Models
All implemented models can be found in the `src/Models.py` file. The `GenericModel` class represents a template that new implemented models must follow in order to be usable. Each architecture must have the following attributes and methods defined:
- self.name: (str) Name of the defined model
- self.weights: weights of the model, used to pull the transforms of pytorch models, (not required)
- self.model: actual model used for predictions. Usually an existing pytorch architecture (e.g. VGG19 or ResNet)
- self.gradients: (None) temporary placeholder to calculate gradients
- self.input_shape: (tuple) model input shape in channels first format
- self.transf_shape: (tuple) model transforms shape in channels first format
- self.transforms: (T.Compose) pytorch transforms to use for preprocessing.

- def self.get_activations(x): all the layers that we want to collect gradients from. 

### Grad-CAM
The file `GradCAM.py` under the *src* folder contains the Grad-CAM implementation, which can be used as below:
```python
grad_cam(model, img_path, true_label, result_dir, exp_name, device):
```
where the following arguments can be defined:
- model:
- img_path:
- true_label:
- result_dir:
- exp_name:
- device:

### Single experiment

### Bulk experiment
(TODO): create a bulk experimenter to get statistics



## Future Work
- batch experimenter


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
<br/>
<div id="tiling_es">
[3].<br/>
Qiu, Hao & Custode, Leonardo & Iacca, Giovanni. (2021). Black-box adversarial attacks using evolution strategies. 1827-1833. 10.1145/3449726.3463137. 
</div>