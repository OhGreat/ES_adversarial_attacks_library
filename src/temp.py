from torch import nn
import numpy as np
from PIL import Image
from GradCAM import grad_cam
from Models import *
from torchvision.models import swin_b, Swin_B_Weights

# model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
# model = ViT_B_16()
# # print(model)
# # exit()
# img_path = 'data/test/tench-1_0.JPEG'
# img = Image.open(img_path)
# img_transf = model.transforms(img).unsqueeze(dim=0)
# preds = model.simple_eval(img_transf)
# print(preds.shape)
# print("Predicted label", preds.argmax(dim=1).item(), "with confidence:", preds.max().item())

# x = img_transf
# print(x.shape)
# x=model.model._process_input(x)
# print(x.shape)
# batch_class_token = model.model.class_token.expand(x.shape[0], -1, -1)
# x = torch.cat([batch_class_token, x], dim=1)
# print(x.shape)
# x = model.model.encoder(x)
# print("encoded:",x.shape)
# # x = x.view(1,-1)
# x = x[:, 0]
# x = model.model.heads(x)
# print("head:", x.shape)
# preds = x
# print("Predicted label", preds[0,1].argmax(dim=0).item(), "with confidence:", torch.softmax(x,dim=1).max().item())
# exit()

print()

# ViT
device="cuda"
img_path = 'data/test/tench-1_0.JPEG'
model = ViT_B_16().to(device)
img = Image.open(img_path).resize(model.transf_shape[1:])
img_transf = model.transforms(img).unsqueeze(dim=0).to(device)

preds = model.simple_eval(img_transf)
print("Predicted label", preds.argmax(dim=1).item(), "with confidence:", preds.max().item())

grad_cam(model, img_path,  result_dir='results/test', exp_name='test', device=device)