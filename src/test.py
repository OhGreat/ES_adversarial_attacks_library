from torch import nn
from PIL import Image
from GradCAM import grad_cam
from Models import VIT_H_14,VIT_B_16
from torchvision.models import swin_b, Swin_B_Weights

# model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
# print(model)
# model = VIT_B_16()
# model = 0
# exit()
img_path = 'data/test/tench-1_0.JPEG'
img = Image.open(img_path)
# img_transf = model.transforms(img).unsqueeze(dim=0)
# preds = model.simple_eval(img_transf)
# print(preds.argmax(dim=1), preds.max())
grad_cam("swin_b", img_path, true_label=0, result_dir='results/temp', exp_name='test')