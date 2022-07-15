import torch
from torch import nn

class GenericModel(nn.Module):
    def __init__(self):
        super(GenericModel, self).__init__()
        self.gradients = None

    def forward(self):
        pass

    def get_activations(self, x):
        """ Should pass input x through all layers before the classification head.
            Must be redefined for each specific model.
        """
        pass

    def activations_hook(self, grad):
        """ Function to call whenever the 
            gradient with respecto to the Tensor is computed.

            In our specific case we simply copy the gradients
        """
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    
class ResNet(GenericModel):
    def __init__(self):
        super(ResNet, self).__init__()
        # define resnet and weights
        from torchvision.models import resnet50, ResNet50_Weights
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.weights)

        # model preprocessing of input images
        self.transforms = self.weights.transforms()

        # placeholder for the gradients
        self.gradients = None

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

    def get_activations(self, x):
        """ Method for the activation exctraction
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class VGG(GenericModel):
    def __init__(self):
        super(VGG, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights

        # get the pretrained VGG19 network
        self.weights = VGG19_Weights.DEFAULT
        self.vgg = vgg19(weights=self.weights)
        
        # all layers before classification head
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool before classification head
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        # it is simply the last sequential block
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
        # model preprocessing for images
        self.transforms = self.weights.transforms()
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the max pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        # pass image to classification head
        x = self.classifier(x)
        return x
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)