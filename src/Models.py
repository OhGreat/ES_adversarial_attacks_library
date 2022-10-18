import torch
from torch import Tensor, nn


class GenericModel(nn.Module):
    def __init__(self):
        super(GenericModel, self).__init__()
        self.name = None
        self.weights = None
        self.model = None
        self.gradients = None
        self.input_shape = None
        self.transf_shape = None
        self.transforms = None
    
    def simple_eval(self, x):
        self.eval()
        x = self.model(x)
        return x.softmax(dim=1)

    def forward(self):
        x = self.model(x)
        return x.softmax(dim=1)

    def grad_cam(self):
        pass

    def get_activations(self, x: Tensor):
        """ Should pass input x through all layers before the classification head.
            Must be redefined for each specific model.
        """
        pass

    def activations_hook(self, grad):
        """ Function to call whenever the 
            gradient with respect to the Tensor is computed.

            In our specific case we simply copy the gradients
        """
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def reshape_gradients(self, x):
        return x


class Swin_b(GenericModel):
    def __init__(self):
        super(Swin_b, self).__init__()
        # model name
        self.name = "swin_b"
        # define model
        from torchvision.models import swin_b, Swin_B_Weights
        self.weights = Swin_B_Weights.IMAGENET1K_V1
        self.model = swin_b(weights=self.weights)

        # expected input shape
        self.input_shape = (3, 224, 224)

        # expected transform shape
        self.transf_shape = (3, 238, 238)

        # image processing transformations
        self.transforms = self.weights.transforms()

    def get_activations(self, x: Tensor):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = x.transpose(2, 3).transpose(1, 2)
        return x

    def grad_cam(self, x: Tensor):
        x = self.get_activations(x)
        h = x.register_hook(self.activations_hook)
        x = self.model.avgpool(x)
        x = x.view((1,-1))
        x = self.model.head(x)
        return x


class ViT_B_16(GenericModel):
    # TODO: fix gradients for Grad-CAM
    def __init__(self):
        super(ViT_B_16, self).__init__()
        # model name
        self.name = "vit_b_16"
        # define model
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1   
        self.model = vit_b_16(weights=self.weights)
        
        # expected input shape
        self.input_shape = (3, 224, 224)

        # transform shape
        self.transf_shape = (3, 256, 256)

        # model image preprocessing transformation
        self.transforms = self.weights.transforms()

    def get_activations(self, x: Tensor):
        x = self.model._process_input(x)
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        for layer in self.model.encoder.layers[:-2]:
            x = layer(x)
        return x
    
    def grad_cam(self, x):
        x = self.get_activations(x)
        h = x.register_hook(self.activations_hook)
        x = self.model.encoder.layers[-2:](x)
        x = self.model.encoder.ln(x)
        x = x[:, 0]
        x = self.model.heads(x)
        return x

    def reshape_gradients(self, x: Tensor):
        x = x[:,1:,:].reshape(x.shape[0], 14, 14, x.shape[2])
        x = x.transpose(2,3).transpose(1,2)
        return x


class ViT_H_14(GenericModel):
    # TODO: fix model
    def __init__(self):
        super(ViT_H_14, self).__init__()
        # model name
        self.name = "vit_h_14"
        # define model
        from torchvision.models import vit_h_14, ViT_H_14_Weights
        self.weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = vit_h_14(weights=self.weights)
        
        # expected input shape
        self.input_shape = (3, 518, 518)

        # transform shape
        self.transf_shape = (3, 518, 518)

        # model image preprocessing transformation
        self.transforms = self.weights.transforms()

    def get_activations(self, x):
        x = self.model.conv_proj(x)
        x = torch.squeeze(x,dim=1)
        x = self.model.encoder(x)
        return x
    
    def grad_cam(self, x):
        x = self.get_activations(x)
        h = x.register_hook(self.activations_hook)
        #x = x.view((1,-1))
        x = self.model.fc(x)
        return x
    

class InceptionV3(GenericModel):
    def __init__(self):
        super(InceptionV3, self).__init__()
        # model name
        self.name = "inception_v3"
        # define model
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=self.weights)

        # expected input shape
        self.input_shape = (3,299,299)

        # transform shape
        self.transf_shape = (3,342,342)

        # model image preprocessing transformation
        self.transforms = self.weights.transforms()

    def get_activations(self, x):
        """ Method for the activation exctraction
        """
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        return x

    def grad_cam(self, x):
        x = self.get_activations(x)
        h = x.register_hook(self.activations_hook)
        x = self.model.avgpool(x)
        x = x.view((1,-1))
        x = self.model.fc(x)
        return x


class ResNet50(GenericModel):
    def __init__(self):
        super(ResNet50, self).__init__()
        # model name
        self.name = "resnet50"
        # define resnet and weights
        from torchvision.models import resnet50, ResNet50_Weights
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights)

        # expected input shape
        self.input_shape = (3,224,224)

        # transform shape
        self.transf_shape = (3,232,232)

        # model preprocessing of input images
        self.transforms = self.weights.transforms()

    def grad_cam(self, x):
        # model before classification head
        x = self.get_activations(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # avg pooling + classification head
        x = self.model.avgpool(x)
        x = x.view((1, -1))
        x = self.model.fc(x)
        return x

    def get_activations(self, x):
        """ Method for the activation exctraction
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class VGG19(GenericModel):
    def __init__(self):
        super(VGG19, self).__init__()
        # model name
        self.name = "vgg19"
        from torchvision.models import vgg19, VGG19_Weights

        # get the pretrained VGG19 network
        self.weights = VGG19_Weights.IMAGENET1K_V1
        self.model = vgg19(weights=self.weights)

        # expected input shape
        self.input_shape = (3,224,224)

        # transform shape
        self.transf_shape = (3,256,256)
        
        # all layers before classification head
        self.features_conv = self.model.features[:36]
        
        # get the max pool before classification head
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        # it is simply the last sequential block
        self.classifier = self.model.classifier
    
        # model preprocessing for images
        self.transforms = self.weights.transforms()
        
    def grad_cam(self, x):
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