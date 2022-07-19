import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import models

TRAIN_DATA_DIR = 'data/face2animetrain'
VALID_DATA_DIR = 'data/face2animetest'
BATCH_SIZE = 256
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 16
EPOCHS = 600
MASKRATE = 87.5
modelname = 'MAE'+str(MASKRATE)+'_'
logfile = 'logs/adamw'+str(MASKRATE)+'.txt'
fo = open(logfile,'w+')
torch.random.manual_seed(888)
#load image data

trans = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR,transform=trans)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)
fo.write('training set loaded: '+str(len(train_dataset))+'\n')
print('training set loaded: '+str(len(train_dataset)))

valid_dataset = datasets.ImageFolder(VALID_DATA_DIR,transform=trans)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)
fo.write('validation set loaded: '+str(len(valid_dataset))+'\n')
print('validation set loaded: '+str(len(valid_dataset)))

#model

MAE = models.MAEViT(maskrate=MASKRATE/100)
MAE.to(device)
optimizer = torch.optim.AdamW(MAE.parameters(), lr=0.0005, betas=(0.9, 0.95))

#train and valid

train_loss_list = []
valid_loss_list = []
for i in range(EPOCHS):
    train_loss = 0
    mb = 0
    for j,data in enumerate(train_dataloader):
        x, y = data
        x = x.to(device)
        optimizer.zero_grad()
        loss,pred,maskid = MAE(x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        mb += 1
        fo.write('epoch={:3d} {:3d}, training_loss={:5f}\n'.format(i,j,loss.item()))
        print('\r epoch={:3d}, training_loss={:5f}'.format(i,loss.item()), flush=True, end='')
    train_loss_list.append(train_loss/mb)
    with torch.no_grad():
        valid_loss = 0
        mb = 0
        for j,data in enumerate(valid_dataloader):
            x, y = data
            x = x.to(device)
            loss,pred,maskid = MAE(x)
            valid_loss += loss.item()
            mb += 1
        valid_loss_list.append(valid_loss/mb)
        fo.write('epoch={:3d}, valid_loss={:5f}\n'.format(i,valid_loss/mb))
        print(' epoch={:3d}, valid_loss={:5f}'.format(i,valid_loss/mb))
    if i%50==49:
        torch.save(MAE,'model/'+modelname+str(i+1)+'.pkl')

plt.plot(train_loss_list, label='train loss')
plt.plot(valid_loss_list, label='valid loss')
plt.legend()
plt.show()
plt.savefig('MAE'+str(MASKRATE)+'.png')
fo.close()