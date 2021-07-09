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


datapath = '../data/pandas_train10.csv'

config = defaults()
config.data_path = datapath
config.regression = True
config.num_workers = 8

config.num_epochs = 50
config.batch_size = 50

config.optimizer = 'adam'
config.activation = 'relu'
config.hidden_sizes = [64,64]

config.lr = 0.01
config.output_regularization = 0
config.l2_regularization = 0
config.decay_rate = 1
config.dropout = 0
config.feature_dropout = 0

train_df = pd.read_csv(datapath)
dataset = NAMDataset(config, data_path = train_df, features_columns = train_df.columns[1:], targets_column='age')

dataloaders = dataset.train_dataloaders()

hidden_sizes_arr = [[], [32], [64, 32], [128, 64, 32]]
lr_arr = [0.0001, 0.001, 0.01, 0.1]
l2_reg_arr = [0, 0.5]

master = []
low = -1

for sizes in hidden_sizes_arr:
    for cur_lr in lr_arr:
        for cur_l2_reg in l2_reg_arr:
            
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

            mae = trainer.test(litmodel, test_dataloaders=dataset.test_dataloaders())
            ##fig2 = plot_nams(litmodel.model, dataset, num_cols= 5)

            if(low == -1 or mae < low):
                master = [sizes, cur_lr, cur_l2_reg]
                low = mae

print(master)
            
            

