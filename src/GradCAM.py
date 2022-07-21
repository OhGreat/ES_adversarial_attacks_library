import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import datasets
from torchvision import transforms as T

from Models import ResNet, VGG, Xception
from os import makedirs
from os.path import exists
from PIL import Image

def grad_cam(model_name, img_path, true_label=None, result_dir="results", exp_name="temp"):

    # choose working model
    if model_name == "resnet50":
        # define model
        model = ResNet()
    elif model_name == "vgg19":
        # define model
        model = VGG()
    elif model_name == "xception_v3":
        model = Xception()
    else:
        exit("Please choose a valid model.")
    print(f"\nModel: {model_name}")
    # set model to evaluation mode
    model.eval()

    # open image
    img = Image.open(img_path)
    # print(f"Initial img shape: {np.array(img).shape}, max: {np.array(img).max()}, min: {np.array(img).min()}")
    # preprocess image
    img = model.transforms(img).unsqueeze(dim=0)
    # print(f"Processed img shape: {img.shape}")
    # predict image
    pred = model.simple_eval(img)
    print(f"Predicted image label: {pred.argmax().item()} with confidence: {torch.max(pred).item()*100} %")
    if true_label is not None:
        print(f"Original label: {true_label} confidence: {pred[:, true_label].item()*100} %")

    # pass image through custom forward pass 
    # to collect gradients of interest
    pred = model(img)
    # backward pass
    pred[:, 0].backward()
    # get the gradients
    gradients = model.get_activations_gradient()
    # pool gradients across channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations until the last convolutional layer
    activations = model.get_activations(img).detach()
    # print("gradients shape:",gradients.shape,"activations shape:",activations.shape)
    # weight the channels by corresponding gradients
    for i in range(gradients.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # apply relu as in original publication
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # make sure results directory exists
    if not exists(result_dir):
        makedirs(result_dir)
    # create heatmap and combined image
    heatmap_img, heated_img = create_images(heatmap=heatmap.numpy(), img_path=img_path)
    # save image and heatmap
    cv2.imwrite(f'./{result_dir}/{exp_name}_combined.jpg', heated_img)
    cv2.imwrite(f'./{result_dir}/{exp_name}_heatmap.jpg', heatmap_img)
    print("Grad-CAM images created succesfully in directory:",result_dir)

def create_images(heatmap, img_path):
    # create superimposed image
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heated_image = heatmap * 0.4 + img

    return heatmap, heated_image


if __name__ == "__main__":
    # VGG models
    # grad_cam(model_name="vgg19",img_path="./data/test/orig_tench.JPEG", 
            # result_dir="results", exp_name="tench" )
    # grad_cam(model_name="vgg19",img_path="./data/test/noisy_input_0.png", 
    #         result_dir="results", exp_name="tench_noise" )

    # Xception models
    # grad_cam(model_name="xception_v3",img_path="./data/test/orig_tench.JPEG", 
    #         result_dir="results/xception", exp_name="tench_xcept" )
    # grad_cam(model_name="xception_v3",img_path="./data/test/shark.JPEG", 
    #         result_dir="results/xception", exp_name="shark_xcept" )
    grad_cam(model_name="xception_v3",img_path="temp_orig.png",
            true_label=0,
            result_dir="results/xception/tench", exp_name="orig" )
    grad_cam(model_name="xception_v3",img_path="temp_atk_img.png",
            true_label=0,
            result_dir="results/xception/tench", exp_name="noise" )
    # grad_cam(model_name="xception_v3",img_path="./data/xception/attack_tench.png", 
            # result_dir="results/xception", exp_name="tench_xcept" )


    # ResNet models
    # grad_cam(model_name="resnet50",img_path="./data/test/orig_tench.JPEG", 
            # result_dir="results", exp_name="tench" )
    pass