import sys
sys.path.append('.')
sys.path.append('src')

from src.attack import adversarial_attack
from src.GradCAM import grad_cam
from src.Models import VGG, ResNet, Xception

if __name__ == "__main__":

    # model = Xception()
    models = [VGG, ResNet, Xception]
    mod_name = ["vgg19", "resnet50", "xception_v3"]
    experiment_base_name = "results/tench_0.05e_1k"
    img = "data/test/tench_0.JPEG"
    true_label = 0
    
    attacks = ["red_channel", "all_channels", "shadow_noise"]

    for mod_idx in range(len(models)):



        for atk_idx in range(len(attacks)):
            experiment_dir = f"{experiment_base_name}/{mod_name[mod_idx]}/{attacks[atk_idx]}"
            print("Curr experiment dir:", experiment_dir)
            adversarial_attack(model=models[mod_idx](), batch_size=16,
                                atk_image=img, atk_mode=atk_idx+1,
                                true_label=true_label, target_label=None,
                                epsilon=0.05, ps=8, os=56,
                                budget=1000, patience=3,
                                verbose=2, result_folder=experiment_dir)

            # gradcam of noisy image
            grad_cam(model_name=mod_name[mod_idx],
                img_path=experiment_dir+"/attack_img.png",
                # img_path=f"data/test/tench_0.JPEG",
                true_label=true_label,
                result_dir=experiment_dir+"/GradCAM", exp_name=f"grad" )
            print("")

        # create original gradCAM
        orig_path = f"{experiment_base_name}/{mod_name[mod_idx]}/GradCAM_orig"
        grad_cam(model_name=mod_name[mod_idx],
                img_path=experiment_dir+"/orig_resized.png",
                true_label=true_label,
                result_dir=orig_path, exp_name=f"orig_gradCAM" )