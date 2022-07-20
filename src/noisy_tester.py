import torch
import numpy as np
from PIL import Image
from Models import Xception
from torchvision import transforms as T

noise = Image.open('noise.png')
print('Noise loaded, shape:', np.array(noise).shape)

img = Image.open('orig.png')
# img = img.resize(np.array(noise).shape[:2])
print('Original image loaded, shape:', np.array(img).shape)

epsilon = 0.05
noise_arr = np.array(noise)
img_arr = np.array(img)
orig_epsilon = epsilon#*img_arr/255.
new_img = (noise_arr*orig_epsilon) + img_arr
new_img = np.uint8(new_img)
Image.fromarray(new_img).save('combined.png')

model = Xception()
model.eval()

# original image prediction
img_transf = model.transforms(Image.fromarray(new_img)).unsqueeze(dim=0)
print(img_transf.shape, img_transf.max(), img_transf.min())
pred = model.simple_eval(img_transf)
print(f"\nPredicted image label: {pred.argmax().item()} with confidence: {torch.max(pred)*100} %\n")

# # process image with noise
# resizer = T.Compose([
#     T.Resize((299,299))
# ])

prod_img = Image.open('temp.png')
prod_img = model.transforms(prod_img).unsqueeze(dim=0)
prod_pred = model.simple_eval(prod_img)
print(f"\nPredicted on created: {prod_pred.argmax().item()} with confidence: {torch.max(prod_pred)*100} %\n")
