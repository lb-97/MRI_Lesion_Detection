import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import random

def positive_pair(data):
    [batch_size,q,h,w]=data.shape
    ret=np.array(data)
    NOISE_R=0.7
    FLIP_R=0.6
    r1=random.random()
    #add random noise
    if r1<NOISE_R:
        ret=ret+np.random.randn(batch_size,q,h,w)*12
    #flip image randomly on all dimensions
    r2=random.random()
    if r2<FLIP_R:
        ret=ret[:,::-1,:,:]
    r3=random.random()
    if r3<FLIP_R:
        ret=ret[:,:,::-1,:]
    if random.random()<0.6 or (r1>NOISE_R and r2>FLIP_R and r1>FLIP_R):
        ret=ret[:,:,:,::-1]
    return torch.as_tensor(ret.copy())

# Different shape on different slices
class cnn_multi_dim(nn.Module):
    def __init__(self,dim=0,output_dim=10):
        super(cnn_multi_dim,self).__init__()
        if dim==1:
            self.conv=nn.Sequential(
                nn.Conv2d(1,8,5,1,0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8,32,3,1,0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32,32,3,1,0),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.out=nn.Linear(2592,output_dim)
        else:
            self.conv=nn.Sequential(
                nn.Conv2d(1,8,5,1,0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8,32,3,1,0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32,32,3,1,0),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.out=nn.Linear(3168,output_dim)    
        self.output_dim=output_dim
            
    def forward(self,x):
            y=x.view([x.shape[1]*x.shape[0],1,x.shape[2],x.shape[3]])
            y=self.conv(y)
            y=y.view(y.size(0),-1)
            y=self.out(y)
            y=y.view(1,y.shape[0],y.shape[1])
            y=nn.AvgPool2d(kernel_size=[x.shape[1],1],stride=[x.shape[1],1])(y)
            y=y.view(y.shape[1],y.shape[2])
            return y

def train(loader,ep,lrate,alpha):
    cnn=[cnn_multi_dim(0),cnn_multi_dim(1),cnn_multi_dim(2)]
    for model in cnn:
        model.to(torch.device("cuda"))
    optimizer=torch.optim.Adam([{"params":cnn[0].parameters()},{"params":cnn[1].parameters()},{"params":cnn[2].parameters()}],lrate)
    for epoch in range(ep):
        losses=[]
        for _,d in enumerate(loader):
            d = d.to(torch.device("cuda"))
            if (_>0):
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

def MeanPool3d(img,kernel):
    s=img.shape
    ret=np.zeros((s[0]//kernel,s[1]//kernel,s[2]//kernel))
    for i in range(0,s[0],kernel):
        for j in range(0,s[1],kernel):
            for k in range(0,s[2],kernel):
                r=img[i:i+kernel,j:j+kernel,k:k+kernel]
                avg=r.mean()
                ret[i//kernel,j//kernel,k//kernel]=avg
    return ret   

def output(nets,images):
    ret=np.zeros((images.shape[0],3,nets[0].output_dim))
    loader=torch.utils.data.DataLoader(dataset=images,batch_size=images.shape[0],shuffle=False)
    with torch.no_grad():
        for _,d in enumerate(loader):
            # Generate slices
            tran_d=[d,d.permute(0,2,1,3),d.permute(0,3,1,2)]
            pred_d=[_,_,_]
            for dim in range(3):
                nets[dim].eval()
                pred_d[dim]=nets[dim](tran_d[dim].float())
                ret[:,dim,:]=pred_d[dim].numpy()
    return ret

if __name__ == "__main__":
    #182 218 182
    print("Entered")
    data=[]
    datasize = 2 # For debug
    for f in os.listdir('data/'):
        if not f.endswith('nii.gz'):
            continue
        print(f"Found {f}...")
        img = nib.load(os.path.join('data/', f))
        print("Finished loading...")
        imgdata=np.array(img.get_fdata())
        print("Finished getting np...")
        imgdata=MeanPool3d(imgdata,2)
        print("Finished pooling...")
        data.append(imgdata)
        if datasize > 0 and len(data) > datasize:
            break

    data=np.array(data)

# TODO: This takes forever!
    print("loaded dataset")

    batch_size=2
    dataloader=torch.utils.data.DataLoader(dataset=data,batch_size=batch_size,shuffle=True)
    cnns=train(dataloader,10,1e-5,40)

