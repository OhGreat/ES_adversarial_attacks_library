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

def grad_cam(model_name, img_path, result_dir="results", exp_name="temp"):

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
    print(f"Model: {model_name}")
    # set model to evaluation mode
    model.eval()

    # open image
    img = Image.open(img_path)
    # preprocess image
    img = model.transforms(img).unsqueeze(dim=0)
    # predict image
    pred = model(img)
    print(f"Predicted image label: {pred.argmax(dim=1)} with confidence: {torch.max(pred)} %")

    # backward pass
    pred[:, 0].backward()
    # get the gradients
    gradients = model.get_activations_gradient()
    print("gradients shape:",gradients.shape)
    # pool gradients across channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations until the last convolutional layer
    activations = model.get_activations(img).detach()
    print("activations shape:",activations.shape)
    # weight the channels by corresponding gradients
    for i in range(512):
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
    # grad_cam(model_name="vgg19",img_path="./data/test/cropped_input_0.png", 
    #         result_dir="results", exp_name="tench" )
    # grad_cam(model_name="vgg19",img_path="./data/test/noisy_input_0.png", 
    #         result_dir="results", exp_name="tench_noise" )
    grad_cam(model_name="xception_v3",img_path="./data/test/noisy_input_0.png", 
            result_dir="results", exp_name="tench_xcept" )
    # ResNet models
    # grad_cam(model_name="resnet50",img_path="./data/test/noisy_input_0.png", 
    #         result_dir="results", exp_name="tench" )
