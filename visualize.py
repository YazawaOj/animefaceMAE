import torch
import matplotlib.pyplot as plt
import models
from PIL import Image
import torchvision.transforms.functional as TF

device = 'cuda:0'
MAE = torch.load('model/MAE75_200.pkl')
MAE.to(device)
x = Image.open('data/face2animetest/testB/39140.jpg').resize((128,128))
ori_img = Image.open('data/face2animetest/testB/39140.jpg').resize((128,128))
x = TF.to_tensor(x).to(device)
ori_img = TF.to_tensor(ori_img).to(device)
x.unsqueeze_(0)
mask_img, pred_img = MAE.show_masks(x)
mask_img.squeeze_(0)
pred_img.squeeze_(0)
plt.subplot(1,3,1)
plt.imshow(torch.einsum('chw->hwc', ori_img).cpu())
plt.subplot(1,3,2)
plt.imshow(torch.einsum('chw->hwc', mask_img).cpu())
plt.subplot(1,3,3)
plt.imshow(torch.einsum('chw->hwc', pred_img).detach().cpu())
plt.savefig('39140.png')

#print(mask_img.shape, pred_img.shape)