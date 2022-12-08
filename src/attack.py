import torch
import time
from PIL import Image
from os import makedirs
from os.path import exists, join
from Models import *
from Evaluation import LogCrossentropy
from EA_sequential.EA import *
from EA_sequential.Mutation import *
from EA_sequential.Selection import *
from EA_sequential.Recombination import *

def adversarial_attack( 
    model: GenericModel,
    atk_image: str,
    atk_mode: int,
    true_label: int = None,
    target_label: int = None,
    es: dict =None,
    ps: int = 8,
    os: int = 56,
    discrete: bool = False,
    epsilon: float = 0.05,
    downsample: bool = None, 
    budget: int = 1000,
    patience: int = 3,
    batch_size: int = 128,
    device: str = None,
    verbose: int = 2,
    result_folder: str = "temp",
) -> None:
    """ 
    Args:
    - model: Model to attack, should be one of the models implemented in the Models.py file
    - atk_image: base image to use for the adversarial attack
    - atk_mode: (int) between 1 and 4 representing the attack method
            - 1: attack only the first channel
            - 2: attack all channels
            - 3: attack all channels with the same noise (shadow approach)
            - 4: one pixel attack methed
    - true_label: real label the image belongs to
    - target_label: targeted label to be used when doing a targeted attack
    - es: dictionary of evolutionary strategy classes
            in the form of {'rec': Recombination(), 'mut': Mutation(), 'sel': Selection()}
    - ps: parent size for the evolutionary algorithm
    - os : offspring size for the evolutionary algorithm
    - epsilon: maximum value of the pixel perturbations
    - downsample: value between (0,1). Factor of which to downsample the input image
    - budget: maximum budget for the attack
    - patience: generations to wait before resetting sigmas if no new best is found
    - batch_size: size of the batch to pass to the model (not yet implemented)
    - device: defines the device to use for the computation. Can be either "cpu" or "cuda". 
    - verbose: debug variable to print information on the terminal
    - result_folder: directory used to save results
    """

    # create results directories if not existant
    if not exists(result_folder):
        makedirs(result_folder)

    # choose best device if not predefined
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # open original image
    orig_img = Image.open(atk_image)
    # save the original image resized to match model transforms shape
    img = orig_img.resize(model.transf_shape[1:])
    img.save(join(result_folder, "orig_resized.png"))
    # process image for model
    img = model.transforms(img).unsqueeze(dim=0)
    print("Preprocessed input image shape:",img.shape)
    # predict label and confidence for initial image
    initial_preds = model.simple_eval(img.to(device))
    print(f"Predicted label {initial_preds.argmax(dim=1).item()} with confidence: {np.round(initial_preds.max().item()*100, 2)}")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(initial_preds[:, target_label].item()*100,3)}%\n")

    # true label control if defined
    if true_label is None:
        true_label = initial_preds.argmax(dim=1).item()
        print(f"True label not defined, using the predicted label {true_label} as ground truth.")
    elif initial_preds.argmax(dim=1).item() != true_label:
        print("WARNING: Initial prediction does not match given true label!")
        exit()
    # if we have a specific target class we make the problem 
    # a maximization for the target class
    label = true_label if target_label is None else target_label
    minimize = True if target_label is None else False

    # EA strategy
    if es == None:
        es = {}
        es['rec'] = GlobalDiscrete()
        es['mut'] = IndividualSigma()
        es['sel'] = CommaSelection()
    elif 'rec' not in es or 'mut' not in es or 'sel' not in es:
        exit('Invalid evolutionary strategy. Make sure keys "rec", "mut" and "sel" are defined.')

    # define evaluation
    eval_ = LogCrossentropy(
        min=minimize,
        atk_mode=atk_mode,
        init_img=orig_img, 
        epsilon=epsilon,
        downsample=downsample,
        label=label,
        model=model,
        batch_size=batch_size,
        device=device,
    )
    
    # get individual size from evaluator
    ind_size = eval_.ind_size

    # create EA
    es = EA(
        minimize=minimize,
        budget=budget,
        patience=patience,
        parents_size=ps, 
        offspring_size=os,
        individual_size=ind_size,
        discrete=discrete,
        recombination=es['rec'],
        mutation=es['mut'],
        selection=es['sel'],
        evaluation=eval_,
        verbose=verbose,
    )
    # run EA
    atk_start = time.time()
    best_noise, _ = es.run()
    atk_end = time.time()
    print(f'Attack time: {np.round((atk_end- atk_start)/60,2)} minutes.')
    
    # save the best found noise as .npy file
    np.save(join(result_folder,"noise"), arr=best_noise)

    # process best found solution over our image
    ind = Population(pop_size=1,ind_size=best_noise.size, mutation=None)
    ind.individuals = np.reshape(best_noise, (1,*best_noise.shape))
    noisy_img = eval_(ind, ret_sol=True)[0]
    
    # save complete attack image as png
    # moveaxis puts the channels in last dimension
    noisy_img = Image.fromarray(np.moveaxis(noisy_img.numpy(),0,2))
    noisy_img.save(join(result_folder, "attack_img.png"))

    # evaluate our final image
    noisy_img = model.transforms(noisy_img).unsqueeze(dim=0)
    pred = model.simple_eval(noisy_img.to(device))
    print(f"Final evaluation pred class: {pred.argmax(axis=1).item()}, confidence: {np.round(pred.max().item()*100,2)}%, confidence original: {np.round(pred[:, true_label].item()*100,2)}%")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(pred[:, target_label].item()*100,2)}%")
