import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nam.config import defaults
from nam.data import FoldedDataset
from nam.data import NAMDataset
from nam.models import NAM
from nam.models import get_num_units
from nam.trainer import LitNAM
from nam.types import Config
from nam.utils import parse_args
from nam.utils import plot_mean_feature_importance
from nam.utils import plot_nams


from nam.data import load_gallup_data

datapath = '../data/pandas_train10.csv'
config = defaults()
config.data_path = datapath
config.regression = True
config.num_workers = 8


def run():
    hparams_run = wandb.init()
    config.update(**hparams_run.config)
    train_df = pd.read_csv(datapath)
    dataset = NAMDataset(config, data_path = train_df, features_columns = train_df.columns[1:], targets_column='age')
    dataloaders = dataset.train_dataloaders()
    model = NAM(
      config=config,
      name="NAM_CPG10",
      num_inputs=len(dataset[0][0]),
      num_units=get_num_units(config, dataset.features),
    )
    for fold, (trainloader, valloader) in enumerate(dataloaders):
        tb_logger = TensorBoardLogger(save_dir=config.logdir,
                                    name=f'{model.name}',
                                    version=f'fold_{fold + 1}')
        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir +
                                            "/{epoch:02d}-{val_loss:.4f}",
                                            monitor='val_loss',
                                            save_top_k=config.save_top_k,
                                            mode='min')
        litmodel = LitNAM(config, model)
        trainer = pl.Trainer(logger=tb_logger,
                           max_epochs=config.num_epochs,
                           checkpoint_callback=checkpoint_callback)
        trainer.fit(litmodel,
                  train_dataloader=trainloader,
                  val_dataloaders=valloader)
        #wandb.log({
        #    "plot_mean_feature_importance": wandb.Image(plot_mean_feature_importance(model, dataset)),
        #    "plot_nams": wandb.Image(plot_nams(model, dataset))
        #})
        
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'activation': {
            'values': ["exu", "relu"]
        },
        "batch_size": {
            'values': [16,64,256]
        },
        "dropout": {
            'min': 0.0,
            'max': 0.99
        },
        "feature_dropout": {
            'min': 0.0,
            'max': 0.99
        },
        "output_regularization": {
            'min': 0.0,
            'max': 0.99
        },
        "l2_regularization": {
            'min': 0.0,
            'max': 0.99
        },
        "lr": {
            'min': 1e-4,
            'max': 0.1
        },
        "hidden_sizes": {
            'values': [[], [32], [64, 32], [128, 64, 32]]
        },
    }
}


sweep_id = wandb.sweep(sweep_config, project="nam")
wandb.agent(sweep_id, function=run)