import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset)

from transformer import MultiViewTransformer

import os
import os.path as osp


class PretrainTransformerDataset(Dataset):
    def __init__(
            self,
            cnn_model: nn.Module,
            image_loader: nn.DataLoader,
            cache_root: str = './transformer_pretraining_dataset/') -> None:
        """Dataset class for pretraining the transformer given cnn

        Args:
            cnn_model (nn.Module): The frozen CNN model
            image_loader (nn.DataLoader): The dataloader of image to feed into CNN
            cache_root (str): The directory to store the activations
        """
        super().__init__()

        # Keep track of data
        self.lst_data_dir = []

        # If the folder doesn't exist, create it
        if not osp.isdir(cache_root):
            os.makedirs(cache_root)

            # First step is to generate a dataset of cnn representations
            with torch.no_grad():
                for image in image_loader:
                    representation = cnn_model(image)

                    # Store the representations
                    name = osp.join(cache_root, len(self.lst_data_dir))
                    torch.save(representation, name)
                    self.lst_data_dir.append(name)
        # If exist, load all the .pt files inside
        else:
            for f in os.listdir(cache_root):
                self.lst_data_dir.append(f)

    def __getitem__(self, idx):
        return torch.load(self.lst_data_dir[idx])

    def __len__(self):
        return len(self.lst_data_dir)

