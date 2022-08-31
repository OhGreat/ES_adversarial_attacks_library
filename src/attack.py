import torch
import time
from PIL import Image
from scipy.ndimage import zoom
from copy import deepcopy
from os import makedirs
from os.path import exists
from torchvision.transforms import Resize
from Models import *
from Evaluation import LogCrossentropy
from EA_components_OhGreat.EA import *
from EA_components_OhGreat.Mutation import *
from EA_components_OhGreat.Selection import *
from EA_components_OhGreat.Recombination import *

def adversarial_attack(model: GenericModel,
                        atk_image: str, atk_mode: int,
                        true_label: int, target_label=None,
                        epsilon=0.05, downsample=None, 
                        ps=8, os=56,
                        budget=1000, patience=3,
                         batch_size=128, device=None,
                        verbose=2, result_folder="temp"):
    """ Parameters:
            - model: Model to attack, should be one of the models implemented in the Models.py file
            - atk_image: base image to use for the adversarial attack
            - atk_mode: (int) between 1 and 4 representing the attack method
                    - 1: attack only the first channel
                    - 2: attack all channels
                    - 3: attack all channels with the same noise (shadow approach)
                    - 4: one pixel attack methed
            - true_label: real label the image belongs to
            - target_label: targeted label to be used when doing a targeted attack
            - epsilon: maximum value of the pixel perturbations
            - ps: parent size for the evolutionary algorithm
            - os : offspring size for the evolutionary algorithm
            - budget: maximum budget for the attack
            - patience: generations to wait before resetting sigma if no new best is found
            - batch_size: size of the batch to pass to the model (not yet implemented)
            - device: defines the device to use for the computation. Can be either "cpu" or "cuda". 
            - verbose: debug variable to print information on the terminal
            - result_folder: directory used to save results
    """

    # create results directories if not existant
    if not exists(result_folder):
        makedirs(result_folder)
        if result_folder[-1] == '/':
            result_folder = result_folder[:-1]

    # choose best device if not predefined
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # open original image
    orig_img = Image.open(atk_image).resize(model.transf_shape[1:])
    # save the original image resized to match model image size
    orig_img.save(f'{result_folder}/orig_resized.png')
    # process image for model
    model = model
    img = deepcopy(orig_img)
    img = model.transforms(img).unsqueeze(dim=0)
    print("Preprocessed input image shape:",img.shape)
    # predict label and confidence for initial image
    initial_preds = model.simple_eval(img.to(device))
    print(f"Predicted label {initial_preds.argmax(dim=1).item()} with confidence: {np.round(initial_preds.max().item()*100, 2)}")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(initial_preds[:, target_label].item()*100,3)}%\n")

    # EA parameters
    recomb = GlobalDiscrete()
    mut = IndividualSigma()
    sel = CommaSelection()

    # define individual size depending on attack
    if atk_mode == "R_channel_only" or atk_mode == "shadow_noise":
        if downsample is not None:
            down_img = zoom(np.array(orig_img), 
                            zoom=(downsample,downsample,1), 
                            order=1)
            ind_size = np.prod(down_img.shape[:2])
            print("Downsampled image shape:", down_img.shape)
        else:
            ind_size = np.prod(model.transf_shape[1:])
    elif atk_mode == "all_channels":  # all channels attack
        if downsample is not None:
            down_img = zoom(np.array(orig_img),
                            zoom=(downsample,downsample,1),
                            order=1)
            ind_size = np.prod(down_img.shape)
            print("Downsampled image shape:", down_img.shape)
        else:
            ind_size = np.prod(model.transf_shape)
    elif atk_mode == "1D_one-pixel":  # 1D one pixel attack
        ind_size = 4  # pixel value, x, y, channel
    elif atk_mode == "3D_one-pixel": # 3D one pixel attack
        ind_size = 5
    else:
        exit("Select a valid attack method.")
    print("Problem dimension (individual size):", ind_size)

    # if we have a specific target class we make the problem a maximization 
    # for the target class
    label = true_label if target_label is None else target_label
    minimize = True if target_label is None else False

    # define evaluation
    eval_ = LogCrossentropy(min=minimize, atk_mode=atk_mode, init_img=orig_img, 
                            epsilon=epsilon, downsample=downsample, label=label,
                            model=model, batch_size=batch_size, device=device)

    # create ES 
    es = EA(minimize=minimize, budget=budget, patience=patience, parents_size=ps, 
            offspring_size=os, individual_size=ind_size, recombination=recomb,
            mutation=mut, selection=sel, evaluation=eval_,verbose=verbose)
    # run ES
    atk_start = time.time()
    best_noise, _ = es.run()
    atk_end = time.time()
    print(f'Attack time: {np.round((atk_end- atk_start)/60,2)} minutes')

    # prerocess original image
    # values between 0-1 and make channels first
    orig_img_norm = (torch.tensor(np.array(orig_img))/255.).permute((2,0,1))
    # process found attack noise
    if atk_mode == "R_channel_only": # one channel attack
        best_noise = torch.tensor(best_noise).clip(-epsilon,epsilon)
        # reshape best found solution to match input image
        if downsample is not None:
            best_noise = best_noise.reshape((1,*down_img.shape[:2]))
            best_noise = Resize(size=model.transf_shape[1:]).forward(best_noise)
        else:
            best_noise = best_noise.reshape(model.transf_shape[1:])
        # create attack image
        noisy_img_arr = orig_img_norm
        noisy_img_arr[0] = torch.add(noisy_img_arr[0], best_noise)
        noisy_img_arr = (noisy_img_arr.clip(0,1)*255).type(torch.uint8)
        print(noisy_img_arr.shape)

    elif atk_mode == "all_channels":  # 3 channels attack
        best_noise = torch.tensor(best_noise).clip(-epsilon,epsilon)
        # reshape best found solution to match input image
        if downsample is not None:
            best_noise = best_noise.reshape((3,*down_img.shape[:2]))
            best_noise = Resize(size=model.transf_shape[1:]).forward(best_noise)
        else:
            best_noise = best_noise.reshape(model.transf_shape)
        # create attack image
        noisy_img_arr = (torch.add(orig_img_norm, best_noise
                        ).clip(0,1)*255).type(torch.uint8)

    elif atk_mode == "shadow_noise":  # noise as shadow on all channels
        best_noise = torch.tensor(best_noise).clip(-epsilon,epsilon)
        # reshape best found solution to match input image
        if downsample is not None:
            best_noise = best_noise.reshape((1,*down_img.shape[:2]))
            best_noise = Resize(size=model.transf_shape[1:]).forward(best_noise)
        else:
            best_noise = best_noise.reshape(model.transf_shape[1:])
        # create attack image
        noisy_img_arr = orig_img_norm
        noisy_img_arr = torch.add(noisy_img_arr, best_noise)
        noisy_img_arr = (noisy_img_arr.clip(0,1)*255).type(torch.uint8)
    
    elif atk_mode == "1D_one-pixel":  # 1D one pixel attack
        best_noise = torch.tensor(best_noise)
        # fix coordinates
        best_noise[1] = (best_noise[1].clip(0,1) * model.input_shape[-2]-1)
        best_noise[2] = (best_noise[2].clip(0,1) * model.input_shape[-1]-1)
        # fix channel
        if best_noise[-1] < 0.33:
            best_noise[-1] = 0
        elif best_noise[-1] >= 0.66:
            best_noise[-1] = 2
        else:
            best_noise[-1] = 1
        # add pixel noise to image
        noisy_img_arr = orig_img_norm
        x = best_noise[1].type(torch.int)
        y = best_noise[2].type(torch.int)
        channel = best_noise[-1].type(torch.int)
        noisy_img_arr[channel,x,y] += best_noise[0]
        noisy_img_arr = (noisy_img_arr.clip(0,1)*255).type(torch.uint8)
    
    elif atk_mode == "3D_one-pixel":  # 3D one pixel attack
        best_noise = torch.tensor(best_noise)
        # fix coordinates
        x = (best_noise[3].clip(0,1) * model.input_shape[-2]-1).type(torch.int)
        y = (best_noise[4].clip(0,1) * model.input_shape[-1]-1).type(torch.int)
        # add noise to each channel of pixel
        noisy_img_arr = orig_img_norm
        noisy_img_arr[0:3,x,y] += best_noise[0:3]
        noisy_img_arr = (noisy_img_arr.clip(0,1)*255).type(torch.uint8)

    # save the best found noise as .npy file
    np.save(file=f'{result_folder}/noise',arr=best_noise)
    # save complete attack image as png
    # moveaxis puts the channels in last dimension
    noisy_img = Image.fromarray(np.moveaxis(noisy_img_arr.numpy(),0,2))
    noisy_img.save(f"{result_folder}/attack_img.png")

    # evaluate our final image
    img_model = model.transforms(noisy_img).unsqueeze(dim=0)
    pred = model.simple_eval(img_model.to(device))
    print(f"Final evaluation pred class: {pred.argmax(axis=1).item()}, confidence: {np.round(pred.max().item()*100,2)}%, confidence original: {np.round(pred[:, true_label].item()*100,2)}%")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(pred[:, target_label].item()*100,2)}%")
