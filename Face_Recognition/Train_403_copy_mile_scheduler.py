import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from torch.optim import lr_scheduler
from ArcMarginProduct import *
import copy
import os
import time
from torch.utils.tensorboard import SummaryWriter
from focal_loss import FocalLoss
from adamp import AdamP
from efficientnet_pytorch import EfficientNet

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.list = os.listdir(self.root_dir)
        self.transform = transform
        self.classes = []

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img = Image.open(self.root_dir + '/' + str(self.list[idx]))
        name = str(self.list[idx]).split('_')[0]
        number = str(self.list[idx]).split('_')[1]
        if self.transform:
            img = self.transform(img)
        label = self.classes.index(name)
        return img, label

classes = os.listdir('High_Resolution_Files')+['June','Kong','Sim']
# Tensorboard : set path
path = './real_pth_folder/efficientNet4_0.001_dark'
#path = '../runs/original_0.001_easy_margin_decading_0.99' 
try :
  shutil.rmtree(path)
except:
  pass
writer = SummaryWriter(path)
# ---------------------- #
device = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')

num_classes = 403
in_channel = 3
batch_size = 128
learning_rate = 0.001
num_epochs = 45

transforms = transforms.Compose([
        transforms.Resize((112,112)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), # 데이터를 PyTorch의 Tensor 형식으로 바꾼다.
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])

#load data
total_dataset=CustomDataset(root_dir='/database/daehyeon/403_dark', transform = transforms)
#total_dataset=CustomDataset(root_dir='/home/capstone_ai1/kong/403',transform = transforms)
total_dataset.classes = classes
total_size = len(total_dataset)
train_size = int(np.ceil(total_size*0.97))
test_size = int(np.floor(total_size*0.03))

train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Tensorboard : dataloader 캡쳐

dataiter = iter(train_loader)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
# matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('face_images', img_grid)

# ---------------------- #

''' define model'''
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=512)
# model = models.resnet50()
# model.fc = nn.Linear(2048  , 512)


# model.load_state_dict(torch.load(path +'.pth'))
model.to(device)
margin = ArcMarginProduct(in_feature=512,out_feature=num_classes,easy_margin=True)
# margin.load_state_dict(torch.load(path+'Margin.pth'))
margin.to(device)
nomargin = ArcMarginForTest(in_feature=512,out_feature=num_classes,easy_margin=True)
nomargin.to(device)
# Tensorboard : network graph 생성

# writer.add_graph(margin, (model(images.to(device)),labels.to(device)))
# writer.close()
classes = tuple([x for x in range(0,num_classes)])
#criterion = FocalLoss(gamma=2, alpha=0.25).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = AdamP([
        {'params': model.parameters(), 'weight_decay': 5e-6},
        {'params': margin.parameters(), 'weight_decay': 5e-6}
    ], lr=learning_rate)
#
# optimizer = torch.optim.Adam([
#     {'params': model.parameters(), 'weight_decay': 5e-6},
#     {'params': margin.parameters(), 'weight_decay': 5e-6}
# # ], lr=learning_rate)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)
m_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.33)
# co_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                        #  T_max=10,
                                        #  eta_min=0)
breaker = False
if __name__ == '__main__':
    total_step = len(train_loader)
    bp = 0

    for epoch in range(num_epochs):
        if breaker == True :
            break
        # if epoch % 20 == 0 and epoch != 0 :
        #     if epoch % 40 == 0 :
        #         print( "40:")
        #         for param in margin.parameters() :
        #                 param.requires_grad= False
        #         for param in model.parameters() :
        #                 param.requires_grad = True
        #     else :
        #         print( "20:")
        #         for param in model.parameters() :
        #                 param.requires_grad = False
        #         for param in margin.parameters() :
        #                 param.requires_grad= True
        #

        for i, (images, labels) in enumerate(train_loader):
            start = time.time()

            # Assign Tensors to Configured Device
            images = images.to(device)
            labels = labels.to(device)

            # Forward Propagation
            logits = model(images)
            outputs,degree,degree2 = margin(logits, labels)

            # Get Loss, Compute Gradient, Update Parameters
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(path)

# tesnorboard : loss 그래프 생성

            if i % 2 == 1:
                writer.add_scalar('intra_degree',
                            degree,
                            epoch * len(train_loader) + i)
                writer.add_scalar('inter_degree',
                            degree2,
                            epoch * len(train_loader) + i)
                writer.add_scalar('training loss',
                            loss.item(),
                            epoch * len(train_loader) + i)
                writer.add_scalar('learning_rate',
                            m_scheduler.get_last_lr()[0],
                            epoch * len(train_loader) + i)
            #if loss < 5/100000:
            #    breaker = True
            # if degree < 23:
            #     torch.save( model.state_dict() , path + '.pth' )
            #     torch.save( margin.state_dict() , path + 'Margin.pth' )
            #     breaker =True
            #     break
# ---------------------- #
            # Print Loss for Tracking Training
            if (i+1) % 1 == 0:
                print("learning_rate:{:.8f}".format(m_scheduler.get_last_lr()[0]))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                nomargin.to(device)
                _, predicted = torch.max(outputs ,1)
                print('Testing data: [Predicted: {} / Real: {}]'.format(predicted, labels))
                nomargin.to('cpu')
            print("걸리는 시간: ", time.time()-start)



        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                nomargin.weight = copy.deepcopy(margin.weight)
                nomargin.to(device)
                outputs = nomargin(logits)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                nomargin.to('cpu')

            writer.add_scalar('accuracy',
            100 * correct / total,
            epoch * len(train_loader) + i)
            print("===========================================================")
            print('Accuracy of the network on the {} test images:\
            {} %'.format(len(test_loader)*batch_size, 100 * correct / total))
            print("===========================================================")
            
            if 100 * correct / total == 100:
                 bp +=1
            else : bp = 0
            if bp == 2:
                breaker = True
        model.train()

        m_scheduler.step()
        # if epoch >= 15:
        #     co_scheduler.step()    
   # if epoch%5 == 0:
        print('pth 모델 저장중...accuracy:{}%'.format(100 * correct / total))
        torch.save(model.state_dict(), path+'.pth')
        torch.save(margin.state_dict(), path+'Margin.pth')
        
        if epoch%30 == 0:
            print('epoch:{} , pth 모델 저장중...accuracy:{}%'.format(epoch,100 * correct / total))
            torch.save(model.state_dict(),path+'{}'.format(epoch)+'.pth')
            torch.save(margin.state_dict(),path+'{}'.format(epoch)+'Margin.pth')

torch.save(model.state_dict(), path+'.pth')
torch.save(margin.state_dict(), path+'Margin.pth')

#
# ct = 0
# for param in model.parameters():
#     ct = ct + 1
#     if ct < 47:
#         param.requires_grad = True
#     else:
#         print(param)
# 