import torch
import numpy as np
from os import makedirs
from os.path import exists
from PIL import Image
from src.attack import adversarial_attack
from src.GradCAM import grad_cam
from src.Models import *


def experiment( atk_img, models, attacks, es=None,
                true_label=None, target_label=None,
                ps=12, os=12*7, budget=1000, 
                epsilon=0.05, downsample=None,
                patience=5, exp_dir="results/temp",
                batch_size=32, device=None,
                verbose=2):
    """Args:
        - atk_img:
        - models: (dict) keys are the names of the folders and the items are the models.
                example: models = {"vgg19": VGG, "resnet50": ResNet, "xception_v3": Xception}
        - attacks: (list) list of (str) attack methods to use.
                example: attacks = ["R_channel_only", "all_channels", "shadow_noise", "1D_one-pixel", "3D_one-pixel"]
        - es: (dict) keys should be 'rec', 'mut', 'sel'. Values should be the functions of the strategy.
                example: es = {'rec': GlobalDiscrete(), 'mut':IndividualSigma(), 'sel': CommaSelection()}
        - true_label: (int) real label whose confidence should be minimized.
        - target_label: (int) value of the label we want to maximize confidence.
        - ps: (int) defines the number of parents.
        - os: (int) defines the number of ossprings per generation.
        - budget: (int) number of maximum fitness function evaluations.
        - epsilon: (float) constraints the noise to an interval of [-e,e].
        - downsample: (float)
        - patience: (int) number of epochs to wait before resetting sigmas, if no new best is found.
        - exp_dir: (str) experiment directory to save results.
        - batch_size: (int) size of the batch to pass to the model.
        - device: (str) defines the torch device to use for computations.
        - verbose: (int) defines the intensity of prints.   
    """
    # create experiment dir if not existant
    if not exists(exp_dir):
        makedirs(exp_dir)
    # results file
    f_name = exp_dir+"/results.txt"
    f = open(f_name, "w")
    f.close()

    # define computing device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing on {device} device.")

    for mod_name in models:
        # prepare model
        model = models[mod_name]().to(device)
        print(f"~~~ Model: {model.name} ~~~")

        # append results of original image
        orig_img = Image.open(atk_img).resize(model.transf_shape[1:])
        orig_img = model.transforms(orig_img).unsqueeze(dim=0)
        with torch.no_grad():
                pred = model.simple_eval(orig_img.to(device))
        print(f"Predicted label {pred.argmax()} with {np.round(pred.max().item()*100,2)}% confidence.")
        # define ground truth label if not defined
        if true_label is None:
            true_label = pred.argmax()
            print("Ground truth label not defined, using argmax of above prediction as true label.")
        with open(f_name, "a") as f:
            f.write(f"{mod_name}\t Label {true_label} original confidence: {np.round(pred[0][true_label].item()*100,2)}\n")

        # create original gradCAM
        orig_path = f"{exp_dir}/{mod_name}/GradCAM_orig"
        grad_cam(model=model, device=device,
                img_path=atk_img,
                true_label=true_label,
                result_dir=orig_path, exp_name=f"orig_gradCAM")
        print("")

        for attack in attacks:
            # define experiment directory
            curr_exp_dir = f"{exp_dir}/{mod_name}/{attack}"
            print("Curr experiment dir:", curr_exp_dir)
            # run attack
            adversarial_attack(model=model, es=es,
                                atk_image=atk_img, atk_mode=attack,
                                 true_label=true_label, target_label=target_label,
                                epsilon=epsilon, downsample=downsample,
                                ps=ps, os=os,
                                budget=budget, patience=patience,
                                batch_size=batch_size, device=device,
                                verbose=verbose, result_folder=curr_exp_dir)

            # gradcam of constructed noisy image
            grad_cam(model=model, device=device,
                img_path=curr_exp_dir+"/attack_img.png",
                true_label=true_label,
                result_dir=curr_exp_dir+"/GradCAM", exp_name=f"grad" )
            print("")

            # write results to file
            img_t = Image.open(curr_exp_dir+"/attack_img.png")
            img_t = model.transforms(img_t).unsqueeze(dim=0)
            with torch.no_grad():
                pred = model.simple_eval(img_t.to(device))
            res_w = f"{mod_name}\t | atk: {attack}\t | confidence orig (label: {true_label}): {np.round(pred[0][true_label].item()*100,2)}\t | pred label: {pred.argmax().item()}, confidence: {np.round(pred[0].max().item()*100,2)}\n"
            with open(f_name, "a") as f:
                f.write(res_w)
        
        with open(f_name, "a") as f:
            f.write("\n")


def bulk_experiment(img_dir, models, attacks, es, ps, os,  budget, epsilon, downsample, patience, exp_dir, batch_size, device, verbose):
    pass