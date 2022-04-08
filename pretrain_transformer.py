import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset)

from transformer import MultiViewTransformer
from training_utils import TrainingModel

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


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer

    parser = ArgumentParser()
    parser.add_argument("--batch_size_generate", type=int, default=20)
    parser.add_argument("--n_worker_generate", type=int, default=128)
    parser.add_argument("--batch_size_pretrain", type=int, default=128)
    parser.add_argument("--n_worker_pretrain", type=int, default=20)
    parser.add_argument("--cache_root",
                        type=str,
                        default="./pretrain_transformer_dataset/")
    parser = TrainingModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # CNN model
    cnn_state_dict_path = "./checkpoint.pt"
    # TODO: Add pretrained CNN model here
    cnn_model = CNN().load_state_dict(cnn_state_dict_path)

    # Load image dataset
    image_dataset = ImageDataset()  # TODO: Add image dataset here
    image_loader = DataLoader(image_dataset,
                              batch_size=args.batch_size_generate,
                              shuffle=True,
                              num_workers=args.n_worker_generate)

    # Instantiate pretraining dataset
    dataset = PretrainTransformerDataset(cnn_model=cnn_model,
                                         image_loader=image_loader,
                                         cache_root=args.cache_root)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size_pretrain,
                        shuffle=True,
                        num_workers=args.n_worker_pretrain)

    # Instantiate training model
    trainer_model = TrainingModel(
        args=args,
        model_args={"Transformer": {}},
        models={"Transformer": MultiViewTransformer},
        model_forward_args={"Transformer": {
            "mask": True
        }},
        model_order=["Transformer"])

    # Instantiate trainer
    # TODO: Add monitoring callbacks here
    trainer = Trainer(accelerator="auto", max_epochs=50)
    trainer.fit(trainer_model, loader)
