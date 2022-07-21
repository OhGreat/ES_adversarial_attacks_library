import torch
import numpy as np
from PIL import Image
from Models import Xception
from torchvision import transforms as T

# noise = Image.open('noise.png')
# print('Noise loaded, shape:', np.array(noise).shape)

# original image
img = Image.open('temp_orig.png')
img_arr = torch.tensor(np.array(img))/255.
print('Original image loaded, shape:', np.array(img).shape)

# noise array
noise_arr = np.load('temp_noise.npy')
noise_arr = torch.tensor(noise_arr.reshape(img_arr.shape))
print(f"noise arr max: {noise_arr.max()}, min: {noise_arr.min()}")

new_img = (torch.add(img_arr, noise_arr).clip(0,1)*255).type(torch.uint8)
print(new_img.shape)

comb_img = Image.fromarray(new_img.numpy())
comb_img.save('temp_combined.png')
print("combined img saved")

model = Xception()
model.eval()
print('lol')

# noised image prediction
# img_transf = model.transforms(Image.fromarray(new_img.numpy())).unsqueeze(dim=0)
img_transf = model.transforms(new_img.permute(2,0,1)).unsqueeze(dim=0)
print(img_transf.shape, img_transf.max(), img_transf.min())
pred = model.simple_eval(img_transf)
print(f"\nPredicted image label: {pred.argmax().item()} with confidence: {torch.max(pred)*100} %\n")

# ES created image prediction
prod_img = Image.open('temp_atk_img.png')
prod_img = model.transforms(prod_img).unsqueeze(dim=0)
prod_pred = model.simple_eval(prod_img)
print(f"\nPredicted on created: {prod_pred.argmax().item()} with confidence: {torch.max(prod_pred).item()*100} %\n")

print("Img difference:", (prod_img - img_transf).sum())