from ml_collections import ConfigDict
from torch.backends.mkl import verbose

from utils.logger_hook import CustomLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
# __all__ = ['get_all_config', 'get_config_trainer']
BATCH_SIZE = 4
NUM_ITER = 5e5
LR_INIT = 2.5e-4



def get_latent_diffusion_config():
    latent_diffusion_configs = ConfigDict()
    latent_diffusion_configs.scale_factor = 0.18215
    latent_diffusion_configs.use_spatial_transformer = False
    latent_diffusion_configs.model_logger_every = 100
    latent_diffusion_configs.diffusion_logger_every = 50
    latent_diffusion_configs.lr_anneal_steps = NUM_ITER
    return latent_diffusion_configs




def get_unet_config():
    unet_configs = ConfigDict()
    unet_configs.image_size = 32
    unet_configs.in_channels = 4
    unet_configs.out_channels = 4
    unet_configs.model_channels = 320
    unet_configs.attention_resolutions = [8, 4, 2, 1]
    unet_configs.num_res_blocks = 2
    unet_configs.channel_mult = [1, 2, 2, 4, 4]
    unet_configs.num_head_channels = 32
    return unet_configs

def get_diffusion_config():
    diffusion_configs = ConfigDict()
    diffusion_configs.linear_start = 0.00085 # 0.0015
    diffusion_configs.linear_end = 0.0120 # 0.0195
    diffusion_configs.timesteps = 1000
    diffusion_configs.beta_schedule = 'linear'
    diffusion_configs.loss_type = 'l2'
    diffusion_configs.first_stage_key = 'image'
    # diffusion_configs.cond_stage_key = 'caption'
    # diffusion_configs.image_size = 64
    diffusion_configs.image_size = 32
    # diffusion_configs.channels = 3
    diffusion_configs.channels = 4
    # diffusion_configs.conditioning_key = 'crossattn'
    diffusion_configs.monitor = 'val / loss_simple_ema'
    diffusion_configs.use_ema = False
    diffusion_configs.clip_denoised = True
    diffusion_configs.l_simple_weight = 1.
    diffusion_configs.use_positional_encodings = False
    diffusion_configs.learn_logvar = False
    diffusion_configs.logvar_init = 0.
    diffusion_configs.parameterization="eps"  # all assuming fixed variance schedules
    return diffusion_configs

def get_first_stage_config():
    first_stage_configs = ConfigDict()
    # first_stage_configs.name = 'autoencoder'
    first_stage_configs.embed_dim = 4
    first_stage_configs.ckpt_path = "pretrained/vae_f_8.ckpt"  # None
    # first_stage_configs.ckpt_path = "../pretrained/vae_f_8.ckpt"  # None
    # first_stage_configs.monitor: val / rec_loss
    first_stage_configs.ddconfig = ConfigDict()
    first_stage_configs.ddconfig.double_z = True
    first_stage_configs.ddconfig.z_channels = 4
    first_stage_configs.ddconfig.resolution = 256
    first_stage_configs.ddconfig.in_channels = 3
    first_stage_configs.ddconfig.out_ch = 3
    first_stage_configs.ddconfig.ch = 128
    first_stage_configs.ddconfig.ch_mult = [1, 2, 4, 4]
    first_stage_configs.ddconfig.num_res_blocks = 2
    first_stage_configs.ddconfig.attn_resolutions = []
    first_stage_configs.ddconfig.dropout = 0.0
    return first_stage_configs

def get_model_config():
    model_configs = ConfigDict()
    model_configs.first_stage_key='image'
    model_configs.condition_stage_key='caption'
    model_configs.condition_stage=None
    model_configs.vae_scale_factor=0.18215
    model_configs.loss_type='l2'
    model_configs.warm_up_steps=6000
    model_configs.lr=9e-5
    model_configs.batch_size=80
    model_configs.diffusion_logger_every=50
    model_configs.with_ema=True

    return model_configs

def get_pl_trainer_config():
    _path = 'checkpoints'
    os.makedirs(_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=_path,  # Save directory
        filename="step-{step}",  # Filename format
        every_n_train_steps=5000,  # Save every 500 steps
        save_top_k=-1,  # Save all checkpoints (set to 1 to keep only the best)
        verbose=True
    )
    trainer_configs = ConfigDict()
    trainer_configs.log_every_n_steps = 10
    trainer_configs.check_val_every_n_epoch = None # Disable epoch-based validation
    # trainer_configs.val_check_interval = 200   # Run validation every 200 steps
    trainer_configs.max_epochs = -1   # disables epoch-based training
    trainer_configs.max_steps = 1e6   # total steps
    trainer_configs.accelerator = 'gpu'   # Use GPU
    trainer_configs.devices = torch.cuda.device_count()   # Use 1 GPU (you can set this to a list of device ids for multi-GPU)
    trainer_configs.callbacks = [CustomLogger('ldm_log'), checkpoint_callback]
    trainer_configs.strategy = "ddp_find_unused_parameters_true" # 'auto'
    return trainer_configs

def get_image_loger_config():
    image_loger_configs = ConfigDict({
                        'n_row':8, 'sample':True,
                        'ddim_steps':None, 'ddim_eta':1.,
                        'plot_reconstruction_rows':False, 'plot_denoise_rows':False,
                        'plot_progressive_rows':False, 'plot_diffusion_rows':False,
                        'return_input':False,
                        'rescale':True, 'log_on':'step', 'clamp':True
                               })
    return image_loger_configs

if __name__ == '__main__':
    pass
    exit(0)