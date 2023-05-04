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
To use the repository a `Python 3` environment is required. The repository has been tested with `Python 3.10.4`. Using a virtual environment is recommended. After creating a virtual environment you can use the following commands to install the required packages:


Optional: follow the Pytorch instructions <a href="https://pytorch.org/">here</a> to install the GPU version.

Install the remaining packages from the `requirements.txt` file provided in the main directory as below:
```bash
pip install -r requirements.txt
```


## Usage


### Adversarial attack
The file `src/attack.py` contains the core function of the repository that allows us to perform adversarial attacks, defined as:
```python
from src.attack import adversarial_attack

def adversarial_attack( 
    model: GenericModel,
    atk_image: str,
    atk_mode: str,
    true_label: int = None,
    target_label: int = None,
    es: dict =None,
    ps: int = 8,
    os: int = 56,
    discrete: bool = False,
    epsilon: float = 0.05,
    downsample: bool = None, 
    budget: int = 1000,
    patienceint: int = 3,
    batch_size: int = 128,
    device: str = None,
    verbose: int = 2,
    result_folder: str = "temp",
) -> None:
```
where:
- `model`: Model to attack, should be one of the models implemented in the Models.py file
- `atk_image`: (str) path to the image to use for the adversarial attack
- `atk_mode`: (int) between 1 and 4 representing the attack method
    - "R_channel": attack only the first channel.
    - "all_channels": attack all channels.
    - "shadow_noise": attack all channels with the same noise (shadow approach).
    - "1D_one-pixel": attack only one channel of one pixel.
    - "3D_one-pixel" attack one pixel on all channels.
- `true_label`: (int) real label the image belongs to
- `target_label`: (int) targeted label to be used when doing a targeted attack
- `epsilon`: (float) maximum value of the pixel perturbations
- `ps`: (int) parent size for the evolutionary algorithm
- `os`: (int) offspring size for the evolutionary algorithm
- `budget`: (int) maximum budget for the attack
- `patience`: (int) generations to wait before resetting sigmas if no new best is found
- `batch_size`: (int) size of the batch to pass to the model (not yet implemented)
- `device`: (str) defines the device to use for the computation. Can be either "cpu" or "cuda". 
- `verbose`: (int) debug variable to print information on the terminal
- `result_folder`: (str) directory used to save results


### Single experiment
The experiment function under the `src/experimenter.py` file can used to run and compare attacks, on various models for a single image . The result statistics are logged in a **results.txt** file, together with the different subfolders for each model and attack, containing the resulting image and grad-CAM visualizations. The definition of the function is the following:
```python
from src.experimenter import experimenter

experiment( atk_img, models, attacks, es=None,
            true_label=None, target_label=None,
            ps=12, os=12*7, budget=1000, 
            epsilon=0.05, downsample=None,
            patience=5, exp_dir="results/temp",
            batch_size=32, device=None,
            verbose=2)
```
where:
- `atk_img`: (str) path of the image to use for the attack.
- `models`: (dict) keys are the names of the folders and the items are the models.
        example: models = {"vgg19": VGG, "resnet50": ResNet, "xception_v3": Xception}
- `attacks`: (list) list of (str) attack methods to use.
        example: attacks = ["R_channel_only", "all_channels", "shadow_noise", "1D_one-pixel", "3D_one-pixel"]
- `es`: (dict) keys should be 'rec', 'mut', 'sel'. Values should be the functions of the strategy.
        example: es = {'rec': GlobalDiscrete(), 'mut':IndividualSigma(), 'sel': CommaSelection()}
- `true_label`: (int) real label whose confidence should be minimized.
- `target_label`: (int) value of the label we want to maximize confidence.
- `ps`: (int) defines the number of parents.
- `os`: (int) defines the number of ossprings per generation.
- `budget`: (int) number of maximum fitness function evaluations.
- `epsilon`: (float) constraints the noise to an interval of [-e,e].
- `downsample`: (float)
- `patience`: (int) number of epochs to wait before resetting sigmas, if no new best is found.
- `exp_dir`: (str) experiment directory to save results.
- `batch_size`: (int) size of the batch to pass to the model.
- `device`: (str) defines the torch device to use for computations.
- `verbose`: (int) defines the intensity of prints.


### Grad-CAM
The file `GradCAM.py` under the *src* folder contains the Grad-CAM implementation, which can be used as below:
```python
from src.GradCAM import grad_cam

grad_cam(model, img_path, true_label, result_dir, exp_name, device)
```
where:
- `model`: model as defined in Models.py
- `img_path`: (str) path to the input image
- `true_label`: (int) true label of the image
- `result_dir`: (str) directory to save results
- `exp_name`: (str) name of the experiment and output image
- `device`: (str) device to use for computations


### Models
All implemented models can be found in the `src/Models.py` file. The `GenericModel` class represents a template that new implemented models must follow in order to be usable. Each architecture must have the following attributes and methods defined:
- `self.name`: (str) Name of the defined model
- `self.weights`: weights of the model, used to pull the transforms of pytorch models, (not required)
- `self.model`: actual model used for predictions. Usually an existing pytorch architecture (e.g. VGG19 or ResNet)
- `self.gradients`: (None) temporary placeholder to calculate gradients
- `self.input_shape`: (tuple) model input shape in channels first format
- `self.transf_shape`: (tuple) model transforms shape in channels first format
- `self.transforms`: (T.Compose) pytorch transforms to use for preprocessing.

- `def self.get_activations(x)`: all the layers that we want to collect gradients from, with x as an input.

## Future Work
- bulk experimenter for multiple images, only to collect statistics.


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