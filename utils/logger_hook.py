from typing import Any
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
from utils.logger import build_logger
from utils.image_logger import ImageLogger
# from configs.train_model_config import get_image_loger_config


class CustomLogger(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_model, _log_date_folder_name = build_logger(log_dir)
        self.log_image = ImageLogger(log_dir, _log_date_folder_name, **{
                        'n_row':8, 'sample':True,
                        'ddim_steps':None, 'ddim_eta':1.,
                        'plot_reconstruction_rows':False, 'plot_denoise_rows':False,
                        'plot_progressive_rows':False, 'plot_diffusion_rows':False,
                        'return_input':False,
                        'rescale':True, 'log_on':'step', 'clamp':True
                               })


    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        _global_step = trainer.global_step
        if _global_step % (self.log_image.frequency//5)== 0:
            _log = {}
            _log.update({'g_step': f'{_global_step:.2e}\t'})
            _log.update({'loss_avg': f'{pl_module.running_loss.avg:.5e}\t'})
            _log.update({'lr': f'{trainer.optimizers[0].param_groups[0]["lr"]:.5e}\t'})
            _log.update({'mem': f'{torch.cuda.max_memory_allocated() / (1024.0 ** 3):.2f}GB'})
            self._log_model(_log)
        if _global_step % (self.log_image.frequency // 5) == 0 and self.log_image.log_on == 'step':
            self.log_image.do_log(pl_module, 'train', batch, _global_step)

    @rank_zero_only
    def _log_model(self, log_dict):
        log_str = "".join([f"{k}: {v}" for k, v in log_dict.items()])
        self.log_model.info(log_str)




