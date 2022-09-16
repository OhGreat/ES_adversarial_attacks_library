import sys
sys.path.append('.')
sys.path.append('src')
from src.experimenter import experiment
from src.Models import *
from EA_components_OhGreat.EA import *
from EA_components_OhGreat.Mutation import *
from EA_components_OhGreat.Selection import *
from EA_components_OhGreat.Recombination import *

if __name__ == "__main__":
    # models = {"resnet50": ResNet50, "inception_v3": InceptionV3, "swin_b":Swin_b, "vit_b_16": ViT_B_16}
    # attacks = ["R_channel_only", "all_channels", "shadow_noise", "1D_one-pixel", "3D_one-pixel"]
    # models = {"swin_b":Swin_b, "vit_b_16": ViT_B_16}

    # exp
    exp_dir="results/temp_2"
    atk_img = "data/test/tench_0.JPEG"
    models = {"vgg19": VGG19, "resnet50": ResNet50, "inception_v3": InceptionV3, "swin_b":Swin_b, "vit_b_16": ViT_B_16 }
    attacks = ["all_channels", "shadow_noise"]
    # attacks = ["R_channel"]
    es = {'rec': GlobalDiscrete(), 'mut':IndividualSigma(), 'sel': CommaSelection()}
    es=None
    experiment( atk_img, models, attacks, es=es,
                true_label=None, target_label=None,
                epsilon=0.05, downsample=None,
                ps=4, os=4*7, budget=100, patience=5,
                batch_size=32, device=None,
                verbose=2, exp_dir=exp_dir)