import os
import torch
import torchvision
import numpy as np
from PIL import Image
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.utilities import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, logger_name, logger_folder,
                 n_row=2,
                 sample=True,
                 ddim_steps=None,
                 ddim_eta=1.,
                 plot_reconstruction_rows=False,
                 plot_denoise_rows=False,
                 plot_progressive_rows=False,
                 plot_diffusion_rows=False,
                 return_input=False,
                 frequency=500,
                 rescale=True,
                 log_on='step',
                 clamp=True):
        super(ImageLogger, self).__init__()
        self.frequency = frequency
        self.save_dir = [logger_name, logger_folder,]
        self.rescale = rescale
        self.log_on = log_on
        self.clamp = clamp
        self.log_image_kwargs = {
                        'n_row':n_row, 'sample':sample,
                        'ddim_steps':ddim_steps, 'ddim_eta':ddim_eta,
                        'plot_reconstruction_rows':plot_reconstruction_rows,
                        'plot_denoise_rows':plot_denoise_rows,
                        'plot_progressive_rows':plot_progressive_rows,
                        'plot_diffusion_rows':plot_diffusion_rows, 'return_input':return_input
        }

    def do_log(self, model, mode, batch, step):
        @rank_zero_only
        def _log_images(split):
            with torch.no_grad():
                images = model.log_image(batch, **self.log_image_kwargs)

            for key in images:  # key could be dict_keys(['inputs', 'reconstructions', 'conditionings', 'diffused_images', 'samples', 'progressive_row'])
                if isinstance(images[key], torch.Tensor):
                    images[key] = images[key].detach().cpu()
                    if self.clamp:
                        images[key] = torch.clamp(images[key], -1., 1.)

            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=2)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                # os.makedirs(os.path.split(path), exist_ok=True)
                os.makedirs(os.path.join(*self.save_dir, f'{split}', 'images', k,), exist_ok=True)
                filename = f'{step:012}.png'
                path = os.path.join(*self.save_dir, f'{split}', 'images', k, filename)
                Image.fromarray(grid).save(path)

        if mode == 'train':
            if self.log_on == 'step' and  step % self.frequency == 0:
                _train_phase = model.model.training
                if _train_phase:
                    model.model.eval()
                _log_images(mode)
                if _train_phase:
                    model.model.train()
        elif mode == 'validation':
            raise NotImplementedError('image logger for validation is not implemented yet!')
            #if self.log_on == 'step' and ((epoch * num_steps_per_epoch) + batch_idx) % (self.frequency//4) == 0:
             #   _log_images(mode)
