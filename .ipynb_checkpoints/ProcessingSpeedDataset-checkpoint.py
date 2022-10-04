import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib


class ProcessingSpeedDataset(Dataset):
    def __init__(self,filename,data_path):
        self.filename=filename
        self.data_path = data_path
        self.dataframe = pd.read_csv('/scratch/vb2183/hcp/data/HCP_S1200_behavior.csv',index_col = 'Subject',usecols=['ProcSpeed_Unadj','Subject'])
    def __getitem__(self,i):
        subject = self.filename[i]
        vol = np.array(nib.load(os.path.join(self.data_path, subject,'fa.nii')).get_fdata()).astype(np.float32)
        label = self.dataframe.at[int(subject[3:]),'ProcSpeed_Unadj']
        return vol, label
    def __len__(self):
        return len(self.filename)
    
    
def loadSubjects(SUBJ_PATH):
    trainSubjects = []
    valSubjects = []
    testSubjects = []
    with open(os.path.join(SUBJ_PATH,"trainList.txt")) as f:
        for line in f.readlines():
            trainSubjects = line.split(' ')
    print("len(trainSubjects): ",len(trainSubjects))
      
    with open(os.path.join(SUBJ_PATH,"valList.txt")) as f:
        for line in f.readlines():
            valSubjects = line.split(' ')
    
    with open(os.path.join(SUBJ_PATH,"testList.txt")) as f:
        for line in f.readlines():
            testSubjects = line.split(' ')
    print("len(testSubjects): ",len(testSubjects))
            
    return trainSubjects, valSubjects, testSubjects
