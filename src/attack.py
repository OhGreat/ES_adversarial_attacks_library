import torch
import time
from PIL import Image
from copy import deepcopy
from os import makedirs
from os.path import exists
from Models import *
from Evaluation import LogCrossentropy
from EA_components_OhGreat.EA import *
from EA_components_OhGreat.Mutation import *
from EA_components_OhGreat.Selection import *
from EA_components_OhGreat.Recombination import *

def adversarial_attack(model: GenericModel, batch_size: int,
                        atk_image: str, atk_mode: int,
                        true_label: int, target_label=None,
                        epsilon=0.05, ps=8, os=56,
                        budget=1000, patience=3,
                        verbose=2, result_folder="temp"):

    # create results directories
    if not exists(result_folder):
        makedirs(result_folder)
        if result_folder[-1] == '/':
            result_folder = result_folder[:-1]

    # model parameters
    model = Xception()
    model.eval()

    # open original image
    orig_img = Image.open(atk_image).resize(model.input_shape[1:])
    # save the original image resized to match model image size
    orig_img.save(f'{result_folder}/orig_resized.png')
    # process image for model
    img = deepcopy(orig_img)
    img = model.transforms(img).unsqueeze(dim=0)
    print("Preprocessed input image shape:",img.shape)
    # predict label and confidence for initial image
    initial_preds = model.simple_eval(img)
    print(f"Predicted label {initial_preds.argmax(dim=1).item()} with confidence: {np.round(initial_preds.max().item()*100, 2)}")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(initial_preds[:, target_label].item()*100,3)}%\n")

    # EA parameters
    recomb = GlobalDiscrete()
    mut = IndividualSigma()
    sel = CommaSelection()
    # define individual size depending on attack
    if atk_mode == 1 or atk_mode == 3:
        ind_size = np.prod(model.input_shape[1:])
    elif atk_mode == 2:
        ind_size = np.prod(model.input_shape)
    else:
        exit("Select a valid attack method.")
    print("Ind shape:", ind_size)

    # if we have a specific target class we make the problem a maximization 
    # for the target class
    label = true_label if target_label is None else target_label
    minimize = True if target_label is None else False

    # define evaluation 
    eval_ = LogCrossentropy(min=minimize, atk_mode=atk_mode, init_img=orig_img, 
                            epsilon=epsilon, label=label,
                            model=Xception(), batch_size=batch_size, device="cuda")

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
    orig_img_norm = torch.unsqueeze((torch.tensor(
                                    np.array(orig_img))/255.
                                    ).permute((2,0,1)), dim=0)
    # process found attack noise
    if atk_mode == 1: # one channel attack
        # reshape best found solution to match input image
        best_noise = torch.tensor(best_noise.reshape(model.input_shape[1:]))
        # create attack image
        noisy_img_arr = orig_img_norm[0]
        noisy_img_arr[0] = torch.add(noisy_img_arr[0], best_noise)
        noisy_img_arr = (noisy_img_arr.clip(0,1)*255).type(torch.uint8)
        print(noisy_img_arr.shape)

    elif atk_mode == 2: # 3 channels attack
        # reshape best found solution to match input image
        best_noise = best_noise.reshape(model.input_shape)
        # create attack image
        noisy_img_arr = (torch.add(orig_img_norm, torch.tensor(best_noise)).clip(0,1)*255).type(torch.uint8)[0]

    elif atk_mode == 3:  # noise as shadow on all channels
        # reshape best found solution to match input image
        best_noise = torch.tensor(best_noise.reshape(model.input_shape[1:]))
        # create attack image
        noisy_img_arr = orig_img_norm[0]
        noisy_img_arr = torch.add(noisy_img_arr, best_noise)
        noisy_img_arr = (noisy_img_arr.clip(0,1)*255).type(torch.uint8)

    # save the best found noise as .npy file
    np.save(file=f'{result_folder}/noise',arr=best_noise)
    # save complete attack image as png
    noisy_img = Image.fromarray(np.moveaxis(noisy_img_arr.numpy(),0,2))
    noisy_img.save(f"{result_folder}/attack_img.png")

    # evaluate our final image
    img_model = model.transforms(noisy_img).unsqueeze(dim=0)
    pred = model.simple_eval(img_model)
    print(f"Final evaluation pred class: {pred.argmax(axis=1).item()}, confidence: {np.round(pred.max().item()*100,2)}%, confidence original: {np.round(pred[:, true_label].item()*100,2)}%")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(pred[:, target_label].item()*100,2)}%")


if __name__ == "__main__":

    model = Xception()
    img = "data/test/orig_tench.JPEG"
    experiment_dir = "results/very_temp"
    a = ["red_channel", "all_channels", "shadow_noise"]

    for i in range(len(a)):
        adversarial_attack(model=model, batch_size=16,
                            atk_image=img, atk_mode=i+1,
                            true_label=0, target_label=None,
                            epsilon=0.01, ps=8, os=56,
                            budget=100, patience=3,
                            verbose=2, result_folder=f'{experiment_dir}/{a[i]}')
        print("")