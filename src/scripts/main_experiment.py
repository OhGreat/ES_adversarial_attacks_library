from os import makedirs
from os.path import exists
import sys
sys.path.append('.')
sys.path.append('src')
import torch
import numpy as np
from PIL import Image
from src.attack import adversarial_attack
from src.GradCAM import grad_cam
from src.Models import VGG, ResNet, Xception

if __name__ == "__main__":

    # choose experiment directory
    experiment_base_name = "results/bird_e0.01_5k"
    if not exists(experiment_base_name):
        makedirs(experiment_base_name)
    # results file
    f_name = experiment_base_name+"/results.txt"
    f = open(f_name, "w")
    f.close()

    # choose attack image
    img = "data/test/bird_11.JPEG"
    true_label = 11
    
    # choose models to attack
    models = [VGG, ResNet, Xception]
    mod_name = ["vgg19", "resnet50", "xception_v3"]
    attacks = ["red_channel", "all_channels", "shadow_noise"]

    for mod_idx in range(len(models)):
        model = models[mod_idx]()
        for atk_idx in range(len(attacks)):
            experiment_dir = f"{experiment_base_name}/{mod_name[mod_idx]}/{attacks[atk_idx]}"
            print("Curr experiment dir:", experiment_dir)
            adversarial_attack(model=model, batch_size=16,
                                atk_image=img, atk_mode=atk_idx+1,
                                true_label=true_label, target_label=None,
                                epsilon=0.01, ps=8, os=56,
                                budget=5000, patience=3,
                                verbose=2, result_folder=experiment_dir)

            # gradcam of noisy image
            grad_cam(model_name=mod_name[mod_idx],
                img_path=experiment_dir+"/attack_img.png",
                # img_path=f"data/test/tench_0.JPEG",
                true_label=true_label,
                result_dir=experiment_dir+"/GradCAM", exp_name=f"grad" )
            print("")

            # write results to file
            img_t = Image.open(experiment_dir+"/attack_img.png")
            img_t = model.transforms(img_t).unsqueeze(dim=0)
            with torch.no_grad():
                pred = model.simple_eval(img_t)
            res_w = f"{mod_name[mod_idx]}\t | atk: {atk_idx }\t | confidence orig (label: {true_label}): {np.round(pred[0][true_label].item()*100,2)}\t | pred label: {pred.argmax().item()}, confidence: {np.round(pred[0].max().item()*100,2)}\n"
            f = open(f_name, "a")
            f.write(res_w)
            f.close()

        # add an empty line to separate models
        f = open(f_name, "a")
        f.write("\n")
        f.close()

        # create original gradCAM
        orig_path = f"{experiment_base_name}/{mod_name[mod_idx]}/GradCAM_orig"
        grad_cam(model_name=mod_name[mod_idx],
                img_path=experiment_dir+"/orig_resized.png",
                true_label=true_label,
                result_dir=orig_path, exp_name=f"orig_gradCAM")