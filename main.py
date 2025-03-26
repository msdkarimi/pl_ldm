import pytorch_lightning as pl
from models.ldm import LatentDM
from configs.train_model_config import get_pl_trainer_config, get_model_config

if '__main__' == __name__:
    model = LatentDM(**get_model_config())
    trainer = pl.Trainer(**get_pl_trainer_config())
    trainer.fit(model)
