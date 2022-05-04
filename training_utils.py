import argparse
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Metric

from typing import Dict, List, Type


class TrainingModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TrainingModel")
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--optimizer", type=str, default="RAdam")
        parser.add_argument("--loss_type",
                            type=str,
                            default="CosineSimilarity")
        return parent_parser

    def __init__(self, args: argparse.ArgumentParser,
                 models: Dict[str, Type[nn.Module]], model_args: Dict[str,Dict],
                 model_forward_args: Dict[str,Dict], model_order: List[str],
                 metrics: Dict[str,Type[Metric]]) -> None:
        """A utility class to wrap models for training using pytorch lightning.

        Args:
            args (argparse.ArgumentParser): The argument to training model.
            models (Dict[str:Type[nn.Module]]): The dictionary of models. Keys are names.
            model_args (Dict[str:Dict]): The keyword argument for instantiating the models.
            model_forward_args (Dict[str:Dict]): The keyword arguments to pass into model each time it's forward called.
            model_order (List[str]): The order to chain the model. Ordered by the name.
            metrics (Dict[str:Type[Metric]]): A list of metrics to be used. Key is the name of metric.
        """
        super().__init__()

        self.models = models
        self.metric = metrics
        self.model_args = model_args
        self.model_order = model_order
        self.model_forward_args = model_forward_args

        # Load the modules
        for name, model in models.items():
            setattr(self, name, model(**args[name]))

        # Save hyperparameters
        log_param_dict = {}
        for model_name, model_arg_dict in model_args.items():
            for arg_name, arg_val in model_arg_dict.items():
                log_param_dict[model_name+'-'+arg_name] = arg_val
        self.save_hyperparameters(log_param_dict)
        
        # Finally log self params too
        self.save_hyperparameters(args)

        # Instantiate loss
        self.loss = getattr(nn, self.hparams.loss_type)()

    def configure_optimizers(self):
        o = getattr(optim, self.hparams.optimizer)
        return o(params=self.parameters(),
                 lr=self.hparams.lr,
                 weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, *args, **kwargs):
        input, target = batch
        for model_name in self.model_order:
            input = getattr(self,
                            model_name)(input,
                                        **self.model_forward_args[model_name])

        # Find the loss
        loss = self.loss(input, target)

        # Find the metrics
        metrics = {n: v(input, target) for n, v in self.metric.items()}
        self.log_dict(metrics)
        return loss
