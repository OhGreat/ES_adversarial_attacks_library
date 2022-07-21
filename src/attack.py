import torch
from PIL import Image
from copy import deepcopy
from Models import *
from Evaluation import LogCrossentropy
from EA_components_OhGreat.EA import *
from EA_components_OhGreat.Mutation import *
from EA_components_OhGreat.Selection import *
from EA_components_OhGreat.Recombination import *

def adversarial_attack(model: GenericModel, batch_size: int,
                        atk_image: str, epsilon, true_label, target_label= None):


    # model parameters
    model = Xception()
    model.eval()

    # open original image
    orig_img = Image.open(atk_image).resize(model.input_shape[:2])
    # save the original image resized to match model image size
    orig_img.save('temp_orig.png')
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
    # define individual size by multiplying all the dimensions
    ind_size = np.prod(img.shape)

    # if we have a pecific target we make the problem a maximization 
    # for the target class
    label = true_label if target_label is None else target_label
    minimize = True if target_label is None else False

    # define evaluation 
    eval_ = LogCrossentropy(min=minimize, init_img=orig_img, 
                            epsilon=epsilon, true_label=label,
                            model=Xception(), batch_size=batch_size, device="cuda")

    # create and run ES 
    es = EA(minimize=minimize, budget=10000, patience=5, parents_size=8, 
            offspring_size=56, individual_size=ind_size, recombination=recomb,
            mutation=mut, selection=sel, evaluation=eval_,verbose=3)
    best_noise, _ = es.run()

    # reshape to match input image
    best_noise = best_noise.reshape((img.shape))#.clip(-epsilon,epsilon)
    # save the best found noise
    np.save(file='temp_noise',arr=best_noise)

    # prerocess original image
    orig_img_norm = torch.unsqueeze((torch.tensor(
                                            np.array(orig_img))/255.
                                            ).permute((2,0,1)), 
                                            dim=0)
    # create attack image
    noisy_img_arr = (torch.add(orig_img_norm, torch.tensor(best_noise)).clip(0,1)*255).type(torch.uint8)

    # save image as png
    noisy_img = Image.fromarray(np.moveaxis(noisy_img_arr[0,:].numpy(),0,2))
    noisy_img.save("temp_atk_img.png")

    # evaluate our final image
    img_model = model.transforms(noisy_img).unsqueeze(dim=0)
    pred = model.simple_eval(img_model)
    print(f"Final evaluation pred class: {pred.argmax(axis=1).item()}, confidence: {np.round(pred.max().item()*100,2)}%, confidence original: {np.round(pred[:, true_label].item()*100,2)}%")
    if target_label is not None:
        print(f"Confidence on targeted class {target_label}: {np.round(pred[:, target_label].item()*100,2)}%")


if __name__ == "__main__":
    model = Xception()
    batches = (128,16)
    img = "temp_orig.png"

    adversarial_attack(model=model, batch_size=batches, atk_image=img,      
                        true_label=0, target_label=2,#target_label=76,
                        epsilon=0.05)