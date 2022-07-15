import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        # define resnet and weights
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.weights)
        print("self",self)
        print()

        # model preprocessing
        self.transforms = self.weights.transforms()

        # placeholder for the gradients
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # model before classification head
        x = self.get_activations(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # avg pooling + classification head
        x = self.resnet.avgpool(x)
        x = x.view((1, 2048))
        x = self.resnet.fc(x)

        return x



    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


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
        # it is simply the last sequential block
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