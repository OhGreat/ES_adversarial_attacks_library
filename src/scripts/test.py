import sys
sys.path.append('.')
sys.path.append('src')
from src.experimenter import experiment
from src.Models import VGG, ResNet, Xception

if __name__ == "__main__":
    # example experiment
    exp_dir="results/temp"
    atk_img = "data/test/bird_11.JPEG"
    models = {"vgg19": VGG, "resnet50": ResNet, "xception_v3": Xception}
    attacks = ["R_channel_only", "all_channels", "shadow_noise", "1D_one-pixel", "3D_one-pixel"]

    experiment( atk_img, models, attacks, 
                true_label=None, target_label=None,
                epsilon=0.05, downsample=None,
                ps=4, os=28, budget=100, patience=5,
                batch_size=32, device=None,
                verbose=2, exp_dir=exp_dir)