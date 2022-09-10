import sys
sys.path.append('.')
sys.path.append('src')
from src.experimenter import experiment
from src.Models import *

if __name__ == "__main__":
    # example experiment
    exp_dir="results/temp_tranformers"
    atk_img = "data/test/tench_0.JPEG"
    # models = {"vgg19": VGG19, "resnet50": ResNet50, "inception_v3": InceptionV3, "swin_b":Swin_b, "vit_b_16": ViT_B_16}
    # attacks = ["R_channel_only", "all_channels", "shadow_noise", "1D_one-pixel", "3D_one-pixel"]
    models = {"swin_b":Swin_b, "vit_b_16": ViT_B_16}
    attacks = ["all_channels", "shadow_noise"]

    experiment( atk_img, models, attacks, 
                true_label=None, target_label=None,
                epsilon=0.05, downsample=None,
                ps=12, os=12*6, budget=1000, patience=5,
                batch_size=32, device=None,
                verbose=2, exp_dir=exp_dir)