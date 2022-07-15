import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        # define resnet and weights
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.weights)

        # model preprocessing
        self.transforms = self.weights.transforms()

        # last conv activation
        print(self.resnet.layer4[2].conv3)
        self.feat_conv = self.resnet.layer4[2].bn3

        # placeholder for the gradients
        self.gradients = None

        self.classifier = self.resnet.parameters()

    def activations_hook(self, grad):
        self.gradients = grad


    def forward(self, x):
        return self.resnet(x)


    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.feat_conv(x)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights

        # get the pretrained VGG19 network
        self.weights = VGG19_Weights.DEFAULT
        self.vgg = vgg19(weights=self.weights)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        print("VGG classifier:",self.classifier)
        
        # placeholder for the gradients
        self.gradients = None
    
        # model preprocessing
        self.transforms = self.weights.transforms()

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)