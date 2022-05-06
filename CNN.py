#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import torch
import matplotlib
import torch.nn as nn
import random
import cv2
import os


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
device = torch.device("cpu")


# In[4]:


def positive_pair(data):
    [batch_size,q,h,w]=data.shape
    ret=data
    ret.to(device)
    NOISE_R=0.7
    FLIP_R=0.6
    r1=random.random()
    #add random noise
    if r1<NOISE_R:
        ret=ret+torch.randn(batch_size,q,h,w)*12
    #flip image randomly on all dimensions
    r2=random.random()
    if r2<FLIP_R:
        ret=torch.flip(ret,[1])
    r3=random.random()
    if r3<FLIP_R:
        ret=torch.flip(ret,[2])
    if random.random()<0.6 or (r1>NOISE_R and r2>FLIP_R and r1>FLIP_R):
        ret=torch.flip(ret,[3])
    return ret


# In[5]:


# Not used
def data_padding(data):
    [batch_size,nx,ny,nz]=data.shape
    maxr=max(nx,ny,nz)
    data2=np.zeros([batch_size,maxr,maxr,maxr])
    for i in range(batch_size):
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    data2[i,x+(maxr-nx)//2,y+(maxr-ny)//2,z+(maxr-nz)//2]=data[i,x,y,z]
    return data2.copy()
    


# In[6]:


# Different shape on different slices
class cnn_multi_dim(nn.Module):
    def __init__(self,dim=0,output_dim=10):
        super(cnn_multi_dim,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,8,5,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8,8))
        )
        self.out=nn.Linear(32*8*8,output_dim)

        self.output_dim=output_dim
            
    def forward(self,x):
            y=x.contiguous().view([x.shape[1]*x.shape[0],1,x.shape[2],x.shape[3]])
            y=self.conv(y)
            y=y.view(y.size(0),-1)
            y=self.out(y)
            y=y.view(1,y.shape[0],y.shape[1])
            y=nn.AvgPool2d(kernel_size=[x.shape[1],1],stride=[x.shape[1],1])(y)
            y=y.view(y.shape[1],y.shape[2])
            return y
    def forward_2(self,x):
            y=x.contiguous().view([x.shape[1]*x.shape[0],1,x.shape[2],x.shape[3]])
            y=self.conv(y)
            y=y.view(y.size(0),-1)
            y=self.out(y)
            print(x.shape,y.shape)
            y=y.view(x.shape[0],x.shape[1],y.shape[1])
            print(y.shape)
            return y


# In[7]:


def train(loader,ep,lrate,alpha,outdim=10):
    cnn=[cnn_multi_dim(0,outdim),cnn_multi_dim(1,outdim),cnn_multi_dim(2,outdim)]
    for net in cnn:
        net.to(device)
    optimizer=torch.optim.Adam([{"params":cnn[0].parameters()},{"params":cnn[1].parameters()},{"params":cnn[2].parameters()}],lrate)
    for epoch in range(ep):
        losses=[]
        for _,d in enumerate(loader):
            if (_>0):
                d.to(device)
                positive=positive_pair(d)
                # Generate slices
                tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
                tran_neg=[negative,negative.permute(0,2,1,3),negative.permute(0,3,1,2)]
                tran_pos=[positive,positive.permute(0,2,1,3),positive.permute(0,3,1,2)]
                pred_d=[_,_,_]
                pred_pos=[_,_,_]
                pred_neg=[_,_,_]
                loss=0
                for dim in range(3):
                    pred_d[dim]=cnn[dim](tran_d[dim].float())
                    pred_pos[dim]=cnn[dim](tran_pos[dim].float())
                    pred_neg[dim]=cnn[dim](tran_neg[dim].float())
                    d1=pred_d[dim]-pred_pos[dim]
                    d2=pred_d[dim]-pred_neg[dim]
                    d1=torch.norm(d1)
                    d2=torch.norm(d2)
                    loss=loss+nn.ReLU()(d1-d2+alpha)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(epoch,_,loss.item())
                losses.append(loss.item())
            negative=d
        print('epoch:',epoch,'loss:',np.array(losses).mean())
        state={0:cnn[0].state_dict(),1:cnn[1].state_dict(),2:cnn[0].state_dict()}
        torch.save(state,'checkpointat{}.pth'.format(epoch))
    return cnn


# In[8]:


#Do not use output
def output(nets,images):
    ret=np.zeros((len(images),3,nets[0].output_dim))
    loader=torch.utils.data.DataLoader(dataset=images,batch_size=len(images),shuffle=False)
    with torch.no_grad():
        for _,d in enumerate(loader):
            # Generate slices
            d.to(device)
            tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
            pred_d=[_,_,_]
            for dim in range(3):
                nets[dim].eval()
                pred_d[dim]=nets[dim](tran_d[dim].float())
                ret[:,dim,:]=pred_d[dim].numpy()
    return ret
    


# In[19]:


#Use output_2 instead
#return: [batch_size, 3, slice_count, feature_count]
def output_2(nets,images):
    ret=np.zeros((len(images),3,max(images[0].shape[0],images[0].shape[1]),nets[0].output_dim))
    loader=torch.utils.data.DataLoader(dataset=images,batch_size=len(images),shuffle=False)
    with torch.no_grad():
        for _,d in enumerate(loader):
            # Generate slices
            d.to(device)
            tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
            pred_d=[_,_,_]
            for dim in range(3):
                nets[dim].eval()
                pred_d[dim]=nets[dim].forward_2(tran_d[dim].float())
                ret[:,dim,0:pred_d[dim].shape[1],:]=pred_d[dim].numpy()
    return ret
    


# In[10]:


import nibabel as nib
#img=nib.load('MPRFlirt_0.nii.gz')
#imgdata=np.array(img.get_fdata())
#print(imgdata.shape)


# In[11]:


#Collect all files with nii.gz 
def scan_files_and_read(path='.'):
    all_files=os.walk(path)
    useful_files=[]
    for (d,b,c) in all_files:
        for cc in c:
            #print(cc)
            if cc.find('nii.gz')>=0:
                useful_files.append(cc)
                
    avgpool=nn.AvgPool3d(kernel_size=2)
    filenames=[]
    for filename in useful_files:
        img=nib.load(filename)
        imgdata=np.array(img.get_fdata())
        imgdata=imgdata[None,:,:,:]
        torchimg=torch.from_numpy(imgdata).to(device)
        out=avgpool(torchimg)
        out=out.squeeze(0)
        
        new_filename=filename[:-7]
        #print(new_filename)
        torch.save(out,new_filename)
        filenames.append(new_filename)
    return filenames


# In[12]:


filenames=scan_files_and_read()


# In[13]:


class Loaded_File():
    def __init__(self,filename):
        self.filename=filename
    def __getitem__(self,i):
        return torch.load(self.filename[i])
    def __len__(self):
        return len(self.filename)
loaded_files=Loaded_File(filenames)
batch_size=4
dataloader=torch.utils.data.DataLoader(dataset=loaded_files,batch_size=batch_size,shuffle=True,drop_last=True)


# In[33]:


#train(dataloader, epoch, learn_rate, alpha, n_feature=10)
cnns=train(dataloader,2,3e-5,40)


# In[39]:


features=output_2(cnns,loaded_files)


# In[46]:


print(features.shape)
features=features-np.mean(features,axis=0)
print(features[1:16,2,44:46,0:5])


# In[ ]:





# In[ ]:




