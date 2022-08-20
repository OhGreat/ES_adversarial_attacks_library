import cv2
import torch
import numpy as np
from PIL import Image
from os import makedirs
from os.path import exists
from copy import deepcopy
from Models import ResNet, VGG, Xception

def grad_cam(model_name, img_path, true_label=None, 
            result_dir="results", exp_name="temp"):

    # choose working model
    if model_name == "resnet50":
        model = ResNet()
    elif model_name == "vgg19":
        model = VGG()
    elif model_name == "xception_v3":
        model = Xception()
    else:
        exit("Please choose a valid model.")
    print(f"\nModel: {model_name}")

    # open image
    img = Image.open(img_path).resize(model.input_shape[1:])
    img_gradCAM = np.array(deepcopy(img))
    # print(f"Initial img shape: {np.array(img).shape}, max: {np.array(img).max()}, min: {np.array(img).min()}")
    # preprocess image
    img = model.transforms(img).unsqueeze(dim=0)
    print(f"Processed img shape: {img.shape}")
    # predict image
    pred = model.simple_eval(img)
    print(f"Predicted image label: {pred.argmax(axis=1).item()} with confidence: {np.round(torch.max(pred).item()*100,2)} %")
    if true_label is not None:
        print(f"Original label: {true_label} confidence: {np.round(pred[:, true_label].item()*100,2)} %")

    # pass image through custom forward pass 
    # to collect gradients of interest
    pred = model.grad_cam(img)
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
    heatmap_img, heated_img = create_images(heatmap=heatmap.numpy(), img=img_gradCAM)
    # save image and heatmap
    cv2.imwrite(f'./{result_dir}/{exp_name}_combined.jpg', heated_img)
    cv2.imwrite(f'./{result_dir}/{exp_name}_heatmap.jpg', heatmap_img)
    print("Grad-CAM images created succesfully in directory:",result_dir)

def create_images(heatmap, img, heatmap_mul=0.3):
    # create superimposed image
    # img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heated_image = (heatmap * heatmap_mul) + (img * (1 - heatmap_mul))

    return heatmap, heated_image


if __name__ == "__main__":

    # experiment_dir = "results/10k_iters"
    # a = ["red_channel", "all_channels", "shadow_noise"]
    
    # for i in range(len(a)):
    #     grad_cam(model_name="xception_v3",
    #             img_path=f"{experiment_dir}/{a[i]}/attack_img.png",
    #             true_label=0,
    #             result_dir=f"{experiment_dir}/{a[i]}/GradCAM", exp_name=f"{a[i]}_atk" )

    grad_cam(model_name="vgg19",
                img_path=f"TEMP/attack_img.png",
                # img_path=f"data/test/tench_0.JPEG",
                true_label=0,
                result_dir=f"TEMP/GradCAM", exp_name=f"aa" )