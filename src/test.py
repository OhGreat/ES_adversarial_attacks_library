from torch import nn
from PIL import Image
from GradCAM import grad_cam
from Models import *
from torchvision.models import swin_b, Swin_B_Weights

# model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
model = ViT_B_16()
print(model)
img_path = 'data/test/tench-1_0.JPEG'
img = Image.open(img_path)
img_transf = model.transforms(img).unsqueeze(dim=0)
preds = model.simple_eval(img_transf)
print(preds.shape)
print("Predicted label", preds.argmax(dim=1).item(), "with confidence:", preds.max().item())

x = img_transf
x = model.model.conv_proj(x)
x = x.view(1, x.shape[2]*x.shape[3], x.shape[1])
# x = x.view(1,x.shape[1], x.shape[2]*x.shape[3])
# x = x.view(x.shape[2], x.shape[3], x.shape[1])
print("conv_proj:",x.shape)
x = model.model.encoder.layers(x)

print("encoded:",x.shape)
# x = x.view(1,-1)
x = model.model.heads(x)
print("head:", x.shape)
preds = x
print("Predicted label", preds.argmax(dim=1).item(), "with confidence:", preds.max().item())
exit()
model = 0
grad_cam("vit_b_16", img_path, true_label=0, result_dir='results/temp', exp_name='test')