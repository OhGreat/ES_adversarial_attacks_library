import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import datasets
from torchvision import transforms as T

from Models import ResNet, VGG
from os import makedirs
from os.path import exists
from PIL import Image

def grad_cam(model_name, img_path):

    # choose working model
    if model_name == "resnet50":
        # define model
        model = ResNet()
    elif model_name == "vgg19":
        # define model
        model = VGG()
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
    # relu as in original publication
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # control that results directory exists
    out_dir = './results'
    if not exists(out_dir):
        makedirs(out_dir)

    # create superimposed image
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    # save image and heatmap
    cv2.imwrite(f'./{out_dir}/map.jpg', superimposed_img)
    cv2.imwrite(f'./{out_dir}/heatmap.jpg', heatmap)
    
if __name__ == "__main__":
    grad_cam("vgg19","./data/test/cropped_input_0.png")
    # grad_cam("vgg19","./data/test/noisy_input_0.png")

    # grad_cam("vgg19","./data/test/2.JPEG")