import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset)
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm
from CNN import PretrainingDataset, cnn_multi_dim, output

from transformer import MultiViewTransformer
from training_utils import TrainingModel

import os
import os.path as osp


class PretrainTransformerDataset(Dataset):
    def __init__(
            self,
            nets: nn.Module,
            image_dataset: Dataset,
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # If the folder doesn't exist, create it
        if not osp.isdir(cache_root):
            os.makedirs(cache_root)

            rets = output(nets, image_dataset, device)
            for x in rets:
                # Store the representations
                name = osp.join(cache_root, str(len(self.lst_data_dir)))
                torch.save(x, name)
                self.lst_data_dir.append(name)
        else:
            for f in os.listdir(cache_root):
                name = os.path.join(cache_root, f)
                self.lst_data_dir.append(name)

    def __getitem__(self, idx):
        return torch.load(self.lst_data_dir[idx])

    def __len__(self):
        return len(self.lst_data_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer

    torch.multiprocessing.set_start_method('forkserver', force=True)    

    parser = ArgumentParser()
    parser.add_argument("--batch_size_generate", type=int, default=8)
    parser.add_argument("--n_worker_generate", type=int, default=8)
    parser.add_argument("--batch_size_pretrain", type=int, default=128)
    parser.add_argument("--n_worker_pretrain", type=int, default=64)
    parser.add_argument("--cache_root",
                        type=str,
                        default="./pretrain_transformer_dataset_cache/")
    parser.add_argument("--imaging_dataset_dir", type=str, default="./data/")
    parser.add_argument("--imaging_dataset_cache_dir", type=str, default="./cached_mri")
    parser.add_argument("--cnn_checkpoint_path", type=str, default="./cnn_checkpoints/checkpointat5.pth")
    parser = TrainingModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # CNN model
    cnn_models = nn.ModuleList([cnn_multi_dim(i) for i in range(3)])
    loaded = torch.load(args.cnn_checkpoint_path)
    for model_idx, model in enumerate(cnn_models):
        model.load_state_dict(loaded[model_idx])

    # Load image dataset
    image_dataset = PretrainingDataset(
        path=args.imaging_dataset_dir, cache_path=args.imaging_dataset_cache_dir  
    )
    # Instantiate pretraining dataset
    dataset = PretrainTransformerDataset(nets=cnn_models,
                                         image_dataset=image_dataset,
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
        model_order=["Transformer"],
        metrics={"MAE": MeanAbsoluteError(), "MAPE": MeanAbsolutePercentageError()}
    )

    # Instantiate trainer
    trainer = Trainer(distributed_backend="ddp", max_epochs=50)
    trainer.fit(trainer_model, loader)
