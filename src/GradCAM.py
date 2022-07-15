import torch
from torch.utils import data
from torchvision import datasets
import numpy as np
from Models import ResNet, VGG
import matplotlib.pyplot as plt

def grad_cam(model):

    if model == "resnet50":
        # define model
        resnet = ResNet()
        # set the evaluation mode
        resnet.eval()

        # define 1 image dataset
        dataset = datasets.ImageFolder(root='./data/', transform=resnet.transforms)
        dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
        # predict image
        img, _ = next(iter(dataloader))
        pred = resnet(img)
        print("Output label:",pred.argmax(dim=1))
        #pred_back = pred[0].backward()

        # backward pass
        pred[:, 0].backward()
        # pull the gradients out of the model
        gradients = resnet.get_activations_gradient()
        print("gradients shape:",gradients.shape)
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # get the activations of the last convolutional layer
        activations = resnet.get_activations(img).detach()
        print("activations shape:",activations.shape)
        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.squeeze())
        #plt.savefig('test.png')

        # create superimposed image
        import cv2
        img = cv2.imread('data/tench/2.JPEG')
        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('./map.jpg', superimposed_img)

    if model == "vgg19":
        # define model
        vgg = VGG()
        # set the evaluation mode
        vgg.eval()

        # define 1 image dataset
        dataset = datasets.ImageFolder(root='./data/', transform=vgg.transforms)
        dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
        # get the image from the dataloader
        img, label = next(iter(dataloader))
        # get the most likely prediction of the model
        pred = vgg(img)

        # backward pass
        pred[:, 0].backward()
        # pull the gradients out of the model
        gradients = vgg.get_activations_gradient()
        print("gradients shape:",gradients.shape)
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # get the activations of the last convolutional layer
        activations = vgg.get_activations(img).detach()
        print("activations shape:",activations.shape)
        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.squeeze())
        #plt.savefig('test.png')

        # create superimposed image
        import cv2
        img = cv2.imread('data/tench/2.JPEG')
        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('./map.jpg', superimposed_img)

        print("Output label:",pred.argmax(dim=1))

if __name__ == "__main__":
    grad_cam("resnet50")
    # grad_cam("vgg19")