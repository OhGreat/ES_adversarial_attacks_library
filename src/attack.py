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
                        atk_image: str, true_label, epsilon=0.02 ):


    # model parameters
    model = Xception()
    model.eval()

    # open original image
    orig_img = Image.open(atk_image).resize(model.input_shape[:2])
    # preprocess image for model
    img = deepcopy(orig_img)
    img = model.transforms(img).unsqueeze(dim=0)
    print("Preprocessed input image shape:",img.shape)

    initial_preds = model.simple_eval(img)
    print(f"Predicted label {initial_preds.argmax(dim=1).item()} with confidence: {initial_preds.max()}\n")

    # define individual size
    ind_size = np.prod(img.shape)


    # EA parameters
    recomb = GlobalDiscrete()
    mut = IndividualSigma()
    sel = CommaSelection()
    # TODO: implement targeted attacks with maximization
    eval_ = LogCrossentropy(min=True, init_img=orig_img, 
                            epsilon=epsilon, true_label=true_label,
                            model=Xception(), batch_size=batch_size, device="cuda")

    # create ES and run 
    es = EA(minimize=True,budget=5000, patience=5, parents_size=8, 
            offspring_size=56, individual_size=ind_size, recombination=recomb,
            mutation=mut, selection=sel, evaluation=eval_,verbose=3)
    best_noise, _ = es.run()


    # reshape to match input image
    best_noise = best_noise.reshape((img.shape)).clip(0,1)
    print(best_noise.shape)
    noise_img = np.uint8(best_noise.reshape(model.input_shape)*255)
    Image.fromarray(noise_img).save("noise.png")

    # scale by epsilon
    orig_epsilon = epsilon*np.moveaxis(np.array(orig_img), 2, 0)/255.
    best_noise = best_noise*orig_epsilon

    orig_img_norm = torch.unsqueeze((torch.tensor(
                                            np.array(orig_img))/255.
                                            ).permute((2,0,1)), 
                                            dim=0)
    noisy_img_arr = (torch.add(torch.tensor(best_noise),orig_img_norm)*255).type(torch.uint8)

    orig_img.save('orig.png')

    # save image as png
    noisy_img = Image.fromarray(np.moveaxis(noisy_img_arr[0,:].numpy(),0,2))
    noisy_img.save("temp.png")

    img_model = model.transforms(noisy_img).unsqueeze(dim=0)
    pred = model.simple_eval(img_model)
    print(f"Final evaluation pred class: {pred.argmax(axis=1).item()}, confidence: {pred.max().item()}, confidence original: {pred[:, true_label].item()}")


if __name__ == "__main__":
    model = Xception()
    batches = (128,16)
    img = "./data/test/orig_tench.JPEG"

    adversarial_attack(model=model, batch_size=batches, 
                        atk_image=img, true_label=0, epsilon=0.05)