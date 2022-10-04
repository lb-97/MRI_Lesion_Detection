#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:

import torch
import torch.nn as nn
import nibabel as nib
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import random
import os

# In[3]:




# In[4]:


def positive_pair(data, device):
    [batch_size,q,h,w]=data.shape
    ret=data
    NOISE_R=0.7
    FLIP_R=0.6
    r1=random.random()
    #add random noise
    if r1<NOISE_R:
        ret=ret+torch.randn(batch_size,q,h,w,device=device)*12
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
    def __init__(self,dim0,output_dim=10):
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
        self.linear=nn.Linear(32*8*8,120)
        self.bn=nn.BatchNorm1d(120)
        self.out=nn.Linear(120,output_dim)

        self.output_dim=output_dim
            
    def forward(self,x):
            y=x.contiguous().view([x.shape[1]*x.shape[0],1,x.shape[2],x.shape[3]])
            y=self.conv(y)
            y=y.view(y.size(0),-1)
            y=self.linear(y)
            y=self.bn(y)
            y=self.out(y)
            y=y.view(1,y.shape[0],y.shape[1])
            y=nn.AvgPool2d(kernel_size=[x.shape[1],1],stride=[x.shape[1],1])(y)
            y=y.view(y.shape[1],y.shape[2])
            return y
    def forward_2(self,x):
            y=x.contiguous().view([x.shape[1]*x.shape[0],1,x.shape[2],x.shape[3]])
            y=self.conv(y)
            y=y.view(y.size(0),-1)
            y=self.linear(y)
            y=self.bn(y)
            y=self.out(y)
            y=y.view(x.shape[0],x.shape[1],y.shape[1])
            return y


# In[7]:


def train(trainLoader,valLoader,ep,lrate,alpha,device,outdim=10,beta1=1,beta2=1):
    cnn=[cnn_multi_dim(0,outdim),cnn_multi_dim(1,outdim),cnn_multi_dim(2,outdim)]
    for net in cnn:
        net.to(device)
    optimizer=torch.optim.Adam([{"params":cnn[0].parameters()},{"params":cnn[1].parameters()},{"params":cnn[2].parameters()}],lrate)
    for epoch in range(ep):
        for loader in [trainLoader, valLoader]:
            losses=[]
            for _, d in enumerate(loader):
                d = d.to(device)
                if (_>0):
                    positive=positive_pair(d, device)
                    # Generate slices
                    tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
                    tran_neg=[negative,negative.permute(0,2,1,3),negative.permute(0,3,1,2)]
                    tran_pos=[positive,positive.permute(0,2,1,3),positive.permute(0,3,1,2)]
                    pred_d=[_,_,_]
                    pred_pos=[_,_,_]
                    pred_neg=[_,_,_]
                    loss=0
                    for dim in range(3):
                        if(loader==valLoader):
                            cnn[dim].eval()
                        pred_d[dim]=cnn[dim](tran_d[dim].float())
                        pred_pos[dim]=cnn[dim](tran_pos[dim].float())
                        pred_neg[dim]=cnn[dim](tran_neg[dim].float())
                        d1=pred_d[dim]-pred_pos[dim]
                        d2=pred_d[dim]-pred_neg[dim]
                        d1=torch.norm(d1)
                        d2=torch.norm(d2)
                        print("Pos and Neg loss:", d1.item(),d2.item())
#                         loss=loss+nn.ReLU()(beta1*d1-beta2*d2+alpha)
                        loss=loss+nn.ReLU()(beta1*d1)
                    if(loader==trainLoader):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    losses.append(loss.item())
                negative=d
            if(loader==trainLoader):
                print('epoch:',epoch,'Train loss:',np.array(losses).mean())
                state={0:cnn[0].state_dict(),1:cnn[1].state_dict(),2:cnn[0].state_dict()}
                torch.save(state, "{}/checkpoint_{}_{}.pt".format(args.checkpoint_path,outdim,epoch))
            else:
                print('epoch:',epoch,'Test loss:',np.array(losses).mean())
                
    return cnn


# In[8]:


#Do not use output
def output(nets,images):
    ret=np.zeros((len(images),3,nets[0].output_dim))
    loader=torch.utils.data.DataLoader(dataset=images,batch_size=len(images),num_workers=16,shuffle=False)
    with torch.no_grad():
        for _,d in enumerate(loader):
            # Generate slices
            d = d.to(device)
            tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
            pred_d=[_,_,_]
            for dim in range(3):
                nets[dim].eval()
                pred_d[dim]=nets[dim](tran_d[dim].float())
                ret[:,dim,:]=pred_d[dim]
    return ret
    


# In[19]:


#Use output_2 instead
#return: [batch_size, 3, slice_count, feature_count]
def output_2(nets,images,device):
    rets = []
    loader=torch.utils.data.DataLoader(dataset=images,batch_size=16,num_workers=16,shuffle=False)
    with torch.no_grad():
        for _,d in enumerate(loader):
            ret = torch.zeros((d.shape[0],3,max(list(images[0].shape)),nets[0].output_dim), device=device)
            # Generate slices
            d = d.to(device)
            tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
            pred_d=[_,_,_]
            for dim in range(3):
                nets[dim].eval()
                pred_d[dim]=nets[dim].forward_2(tran_d[dim].float())
                ret[:,dim,0:pred_d[dim].shape[1],:]=pred_d[dim]
            for x in ret:
                rets.append(x)
    return rets


def output_single(nets,image):
    ret = torch.zeros((image.shape[0],3,max(list(image.shape[1:])),nets[0].output_dim), device=image.device)
    tran_d=[image,image.permute(0,2,1,3),image.permute(0,3,1,2)]
    for dim in range(3):
        pred_temp=nets[dim].forward_2(tran_d[dim].float())
        ret[:,dim,:pred_temp.shape[1],:]=pred_temp
    return ret



# In[10]:


#img=nib.load('MPRFlirt_0.nii.gz')
#imgdata=np.array(img.get_fdata())
#print(imgdata.shape)


# In[11]:


#Collect all files with nii.gz 
def scan_files_and_read(path='data/', cache_path='cached_mri/', device='cpu'):
    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)
        all_files=os.walk(path)
        useful_files=[]
        for (d,b,c) in all_files:
            for cc in c:
                #print(cc)
                if cc.find('nii.gz')>=0:
                    useful_files.append(cc)

        avgpool=nn.AvgPool3d(kernel_size=2)
        filenames=[]
        for filename in tqdm(useful_files, desc="Caching dataset..."):
            img=nib.load(os.path.join(path, filename))
            imgdata=np.array(img.get_fdata())
            imgdata=imgdata[None,:,:,:]
            torchimg=torch.from_numpy(imgdata).to(device)
            out=avgpool(torchimg)
            out=out.squeeze(0)

            new_filename = os.path.join(cache_path, filename[:-7])
            #print(new_filename)
            torch.save(out, new_filename)
            filenames.append(new_filename)
    else:
        filenames = []
        for f in os.listdir(cache_path):
            filenames.append(f)
    return filenames


# In[12]:



# In[13]:
class Loaded_File(Dataset):
    def __init__(self,filename,data_path):
        self.filename=filename
        self.data_path = data_path
    def __getitem__(self,i):
        return np.array(nib.load(os.path.join(self.data_path, self.filename[i],'fa.nii')).get_fdata()).astype(np.float32)
    def __len__(self):
        return len(self.filename)

class PretrainingDataset(Dataset):
    def __init__(self, path='data/', cache_path='cached_mri/'):
        self.filename = scan_files_and_read(path, cache_path)
        self.path = path
        self.cache_path = cache_path
    def __getitem__(self, idx):
        return torch.load(os.path.join(self.cache_path, self.filename[idx]))
    def __len__(self):
        return len(self.filename)

def loadSubjects(SUBJ_PATH):
    trainSubjects = []
    valSubjects = []
    with open(os.path.join(SUBJ_PATH,"trainList.txt")) as f:
        for line in f.readlines():
            trainSubjects = line.split(' ')
    print("len(trainSubjects): ",len(trainSubjects))
      
    with open(os.path.join(SUBJ_PATH,"valList.txt")) as f:
        for line in f.readlines():
            valSubjects = line.split(' ')
            
    return trainSubjects, valSubjects

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=3e-3)
    parser.add_argument("--beta1", type=float, default=1)
    parser.add_argument("--beta2", type=float, default=1)
#     parser.add_argument("--raw_dataset", type=str, default='./data/')
#     parser.add_argument("--dataset_cache", type=str, default='./cached_mri/')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
        
#     filenames=scan_files_and_read(args.raw_dataset, args.dataset_cache)
    SRC_PATH = "/scratch/vb2183/hcp/data/metrics_postprocessed"
    SUBJ_PATH = os.path.join("/scratch/vb2183/hcp/","MRI_Lesion_Detection","subjects")
    trainSubjects , valSubjects = loadSubjects(SUBJ_PATH)
    
            
    batch_size=10
    trainLoader=torch.utils.data.DataLoader(dataset=Loaded_File(trainSubjects, SRC_PATH),batch_size=batch_size,shuffle=True,drop_last=True)
    valLoader=torch.utils.data.DataLoader(dataset=Loaded_File(valSubjects, SRC_PATH),batch_size=batch_size,shuffle=True,drop_last=True)
    
    epochs = args.epochs if args.pretrain else 0
#     cnns=train(trainLoader,valLoader,epochs,1e-3,3e-5, device, outdim=args.hidden_size)
    cnns=train(trainLoader,valLoader,epochs,1e-3,args.alpha,device, outdim=args.hidden_size,beta1=args.beta1,beta2=args.beta2)
    
    # Store in CPU
#     for c in cnns:
#         c.to('cpu')
#     state={0:cnns[0].state_dict(),1:cnns[1].state_dict(),2:cnns[0].state_dict()}
#     torch.save(state, args.checkpoint_path)

