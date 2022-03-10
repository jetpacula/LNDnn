#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torchvision.models as models
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from skimage.color import rgba2rgb
import numpy
import segmentation_models_pytorch as smp


# #раскомментируйте ниже для скачивания датасета

# import boto3
# from botocore import UNSIGNED
# from botocore.client import Config
# s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
# 
# with open('data.zip', 'wb') as f:
#     s3.download_fileobj('intelinair-data-releases', 'longitudinal-nutrient-deficiency/Longitudinal_Nutrient_Deficiency.zip', f)

# unzip -qq ./data.zip # распаковка датасета

# дашборд для построения графиков и сравнения моделей / loss функций

# In[7]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[8]:


from torch.utils.tensorboard import SummaryWriter # TensorBoard 
tb = SummaryWriter(comment='semantic segmentation')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device


# In[6]:


from torch.utils.data import Dataset

import cv2
import PIL
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_labels = [x[0]+"/nutrient_mask_g0.png" for x in os.walk(datasetDir)][1:]
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        # конвертируем датасет в 3-слойное изображение
        image = PIL.Image.open(os.path.join(datasetDir, "field_"+"{:03d}".format(idx+1)+"/image_i0.png")).convert('RGB') 
        #конвертируем маску в 1-слойное изображение
        label =  PIL.Image.open(os.path.join(datasetDir, "field_"+"{:03d}".format(idx+1)+"/nutrient_mask_g0.png")).convert('1')
        if self.transform:
            image = self.transform(image) 

            label = self.transform(label)
        return image, label


# In[7]:


import os
datasetDir = os.path.join(os.getcwd(),'Longitudinal_Nutrient_Deficiency')
mydataset = CustomDataset(img_dir = datasetDir, transform = transforms.Compose([transforms.Resize(256)
    , transforms.CenterCrop(256),transforms.ToTensor()])) # приводим к единому размеру и конвертируем в тензор


# In[8]:


from sklearn.model_selection import train_test_split
training_data, testing_data = train_test_split(mydataset, test_size=0.2, random_state=25) #разделяем


# In[9]:


train_loader =  torch.utils.data.DataLoader(training_data, 
                                          batch_size=4, 
                                          shuffle=True, 
                                          num_workers=1)
    
test_loader = torch.utils.data.DataLoader(testing_data, 
                                          batch_size=4, 
                                          shuffle=True, 
                                          num_workers=1)


# In[10]:


import time
def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=20,quantizing=False):
    start = time.time()
    model = model.to(device)

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl
            running_loss = 0.0
            running_acc = 0.0
            step = 0            
            for x, y in dataloader:
                if quantizing:
                    x = torch.quantize_per_tensor(x, 0.1, 10, torch.quint8)
                    y = torch.quantize_per_tensor(y, 0.1, 10, torch.quint8)
                x = x.to(device)
                y = y.to(device)
                step += 1
                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                acc = acc_fn(outputs, y)
                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cpu()).float().mean()


# In[12]:


# IoU метрика для сегментации 
def numpy_iou(outs,labels, threshold=0.5):
    outs = (outs > threshold)
    intersection = numpy.logical_and(outs.cpu().detach().numpy(), labels.cpu().detach().numpy())
    union = numpy.logical_or(outs.cpu().detach().numpy(), labels.cpu().detach().numpy())
    iou_score = numpy.sum(intersection) / numpy.sum(union)
    return iou_score


# In[16]:


#loss-функция

loss_fn = smp.losses.TverskyLoss("binary", alpha=2, beta=5, gamma=5)

#cnn модель
model = smp.DeepLabV3Plus(
    encoder_name="timm-efficientnet-b1",       
    encoder_weights="imagenet",    
    in_channels=3,                  
    classes=1,                      
)


# In[ ]:


# оптимизатор
opt = torch.optim.AdamW(model.parameters(),lr=0.001) 

#обучаем модель
train_loss, valid_loss = train(model,train_loader,test_loader,loss_fn,opt,numpy_iou,epochs=40,quantizing=False)
torch.save(model.state_dict(), "tversky_loss_SMOL_40_epochs_iou_DeepLab_NOTquantized.h5")


# In[ ]:


backend = "qnnpack" # драйвер для статической квантизации
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False) 
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

torch.save(model_static_quantized.state_dict(), "tversky_loss_SMOL_40_epochs_iou_DeepLab_NOTquantized.h5")


# In[ ]:


tb.close()
get_ipython().run_line_magic('tensorboard', '--logdir=runs')


# In[46]:


#сравниваем размер квантизованной / неквантизованной модели

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')
print('original model size')
print_model_size(model)
print('quantized model size')
print_model_size(model_static_quantized)

