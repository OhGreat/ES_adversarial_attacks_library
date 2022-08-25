import sys
sys.path.append('.')
sys.path.append('src')
import torch
import numpy as np
from os import makedirs
from os.path import exists
from PIL import Image
from src.attack import adversarial_attack
from src.GradCAM import grad_cam
from src.Models import VGG, ResNet, Xception

if __name__ == "__main__":
    ##########################################################################
    # TODO: Tune for every experiment
    ##########################################################################
    # choose experiment directory
    experiment_base_name = "results/tench_e0.05-4k_iters"
    if not exists(experiment_base_name):
        makedirs(experiment_base_name)
    # results file
    f_name = experiment_base_name+"/results.txt"
    f = open(f_name, "w")
    f.close()

    # TODO: Tune for each experiment
    # choose attack image
    img = "data/test/tench_0.JPEG"
    true_label = 0
    
    # TODO: make as dictionaries models and their dir names
    # choose models to attack
    models = [VGG, ResNet, Xception]
    mod_name = ["vgg19", "resnet50", "xception_v3"]
    attacks = ["R_channel_only", "all_channels", "shadow_noise", "1D_one-pixel", "3D_one-pixel"]

    # define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing on {device} device.")
    #########################################################################

    for mod_idx in range(len(models)):
        # prepare model
        model = models[mod_idx]().to(device)
        for attack in attacks:
            # define experiment directory
            experiment_dir = f"{experiment_base_name}/{mod_name[mod_idx]}/{attack}"
            print("Curr experiment dir:", experiment_dir)
            # run attack
            adversarial_attack(model=model,
                                atk_image=img, atk_mode=attack,
                                 true_label=true_label, target_label=None,
                                epsilon=0.05, ps=8, os=56,
                                budget=4000, patience=5,
                                batch_size=16, device=device,
                                verbose=2, result_folder=experiment_dir)
            # gradcam of constructed noisy image
            grad_cam(model_name=mod_name[mod_idx],
                img_path=experiment_dir+"/attack_img.png",
                true_label=true_label,
                result_dir=experiment_dir+"/GradCAM", exp_name=f"grad" )
            print("")

            # write results to file
            img_t = Image.open(experiment_dir+"/attack_img.png")
            img_t = model.transforms(img_t).unsqueeze(dim=0)
            with torch.no_grad():
                pred = model.simple_eval(img_t.to(device))
            res_w = f"{mod_name[mod_idx]}\t | atk: {attack}\t | confidence orig (label: {true_label}): {np.round(pred[0][true_label].item()*100,2)}\t | pred label: {pred.argmax().item()}, confidence: {np.round(pred[0].max().item()*100,2)}\n"
            f = open(f_name, "a")
            f.write(res_w)
            f.close()

        # create original gradCAM
        orig_path = f"{experiment_base_name}/{mod_name[mod_idx]}/GradCAM_orig"
        grad_cam(model_name=mod_name[mod_idx],
                img_path=experiment_dir+"/orig_resized.png",
                true_label=true_label,
                result_dir=orig_path, exp_name=f"orig_gradCAM")

        # append results of original image
        orig_img = Image.open(experiment_dir+"/orig_resized.png")
        orig_img = model.transforms(orig_img).unsqueeze(dim=0)
        with torch.no_grad():
                pred = model.simple_eval(orig_img.to(device))
        f = open(f_name, "a")
        f.write(f"Label {true_label} original confidence: {np.round(pred[0][true_label].item()*100,2)}\n\n")
        f.close()
