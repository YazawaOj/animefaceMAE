import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

#torch.random.manual_seed(888)

device = 'cuda:1'
MAE = torch.load('model/P8MAE75_600.pkl') #load model
MAE.to(device)
IMAGEFILE = 'data/face2animetest/testB/1941088.jpg'
MASKRATE1 = 75 #masking rate of the original model
MASKRATE2 = 75 #masking rate of the visualization
MAE.maskrate = MASKRATE2/100
x = Image.open(IMAGEFILE).resize((128,128))
ori_img = Image.open(IMAGEFILE).resize((128,128))
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
plt.savefig('vis_{:.1f}_{:.1f}.png'.format(MASKRATE1,MASKRATE2))
