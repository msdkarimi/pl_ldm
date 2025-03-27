from contextlib import contextmanager
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from einops import rearrange, repeat
from models.autoencoder import AutoencoderKL
from models.diffusion import GaussianDiffusion
from models.openai_model import UNetModel
from configs.train_model_config import get_first_stage_config, get_diffusion_config, get_unet_config
from torch.optim.lr_scheduler import LambdaLR
from data.data_loader import DareDataset
from utils.utils import image_transform, make_grid, noise_like, AverageMeter
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import grad_norm
from tqdm import tqdm
from models.ema import LitEma

class LatentDM(pl.LightningModule):
    def __init__(self,
                 first_stage_key='image',
                 condition_stage_key='caption',
                 condition_stage=None,
                 vae_scale_factor=0.18215,
                 loss_type='l2',
                 warm_up_steps=5000,
                 lr=1e-4,
                 batch_size=1,
                 diffusion_logger_every=50,
                 with_ema=False,
                 ):
        super().__init__()
        self.model = UNetModel(**get_unet_config())
        self.vae = AutoencoderKL(**get_first_stage_config())
        self.diffusion = GaussianDiffusion(**get_diffusion_config())
        self.first_stage_key = first_stage_key
        self.condition_stage_key = condition_stage_key
        self.condition_stage = condition_stage
        self.vae_scale_factor = vae_scale_factor


        self.loss_type = loss_type
        self.warm_up_steps = warm_up_steps
        self.lr = lr

        self.batch_size = batch_size
        self.diffusion_logger_every = diffusion_logger_every

        self.with_ema = with_ema
        if self.with_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")


        self.running_loss = AverageMeter()

    def forward(self, x, t, context):
        return self.model(x, t, context=context)

    def _prepare_inputs(self, batch):
        x = batch[self.first_stage_key]
        z = self.encode_image(x)
        c = self.encode_condition(batch[self.condition_stage_key]) if self.condition_stage is not None else None
        noise = torch.randn_like(z).to(z.device)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        z_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
        return z_noisy, t, noise, c

    def training_step(self, batch, batch_idx):
        noisy_input, t, target_noise, c = self._prepare_inputs(batch)
        pred = self.forward(noisy_input, t, context=c)
        simple_loss = self.compute_loss(pred, target_noise, loss_type=self.loss_type).mean()
        self.running_loss.update(simple_loss.item())
        _phase = 'train' if self.model.training else 'val'
        self.log(f'{_phase}_loss', simple_loss.item())
        return simple_loss

    def on_after_backward(self) -> None:
        norms = grad_norm(self, norm_type=2)  # 2-norm by default
        self.log_dict(norms, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        for param in self.parameters():
            total_norm += torch.norm(param, p=2) ** 2

        self.log('param_norm', total_norm.sqrt().item(), on_step=True, on_epoch=False)
        for opt in self.trainer.optimizers:
            self.log('lr', opt.param_groups[0]['lr'], on_step=True, on_epoch=False)


    def compute_loss(self, pred, target, loss_type='l2'):
        if loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
        elif loss_type == 'l1':
            loss = (target - pred).abs()
        else:
            raise NotImplementedError
        loss = loss.mean([1, 2, 3])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.warm_up_steps:
            def lr_lambda(current_step):
                if current_step < self.warm_up_steps:
                    return float(current_step) / float(max(1, self.warm_up_steps))
                return 1.0  # After warmup, keep the learning rate constant
            scheduler = {
                'scheduler': LambdaLR(optimizer, lr_lambda=lr_lambda),
                'interval': 'step', # Update at every step
                'frequency': 1
            }
            return [optimizer], [scheduler]
        return optimizer


    def on_train_start(self):
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False  # Freeze vae

    @torch.no_grad()
    def encode_image(self, x):
        z = self.vae.encode(x).sample().detach()
        if self.vae_scale_factor:
            z *= self.vae_scale_factor
        return z
    @torch.no_grad()
    def decode_latent(self, z):
        if self.vae_scale_factor:
            z = 1. / self.vae_scale_factor * z
        return self.vae.decode(z)
    @torch.no_grad()
    def encode_condition(self, c):
        if c is None:
            return None
        return self.condition_stage(c)


    def log_image(self, batch,
                  n_row=4, sample=True,
                  ddim_steps=None, ddim_eta=1.,
                  plot_denoise_rows=False,
                  plot_progressive_rows=False,
                  plot_diffusion_rows=False,
                  plot_reconstruction_rows=False,
                  return_input=False,):
        use_ddim = ddim_steps is not None
        _log = dict()

        x, c = batch['image'], batch['caption']
        _batch_size = x.shape[0]
        _samples = min(_batch_size, n_row)
        x = x.cuda() # todo update here
        z = self.encode_image(x)
        if self.condition_stage:
            c = self.encode_condition(c)
        else:
            c = None

        if return_input:
            _log.update({'input': x})
        if plot_reconstruction_rows:
            reconstruction = self.decode_latent(z)
            _log.update({'reconstruction': reconstruction})
        if plot_diffusion_rows:
            _diffused_images = list()
            z_start = z[:_samples]
            for t in range(self.diffusion.num_timesteps):
                if t % self.diffusion_logger_every == 0 or t == self.diffusion.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=_samples)
                    t = t.cuda().long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.diffusion.q_sample(x_start=z_start, t=t, noise=noise)
                    _diffused_images.append(self.decode_latent(z_noisy))
            diffusion_row = torch.stack(_diffused_images)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            # _log["diffused_images"] = diffusion_grid
            _log.update({'diffused_images': diffusion_grid})
        if sample:
            # todo add later ema model
            samples, z_denoise_row = self.sample_log(c, _samples) # z_denoise_row is intermediates from T to 0
            x_samples = self.decode_latent(samples)
            # _log["samples"] = x_samples
            _log.update({'samples': x_samples})
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                # _log["denoise_row"] = denoise_grid
                _log.update({'denoise_row': denoise_grid})

        if plot_progressive_rows:
            pass
            # todo to be implemented
        return _log

    def _get_denoise_row_from_list(self, samples, desc=''):
        # for visualization purposes only
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_latent(zd))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim=None, ddim_steps=None, **kwargs):
        # todo ddim to be implemented
        # if ddim:
        #     ddim_sampler = DDIMSampler(self)
        #     shape = (self.channels, self.image_size, self.image_size)
        #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
        #                                                  shape, cond, verbose=False, **kwargs)
        #
        # else:
        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def sample(self, cond, batch_size, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, x0=None, shape=None):

        if shape is None:
            shape = (batch_size, self.model.in_channels, self.model.image_size, self.model.image_size)

        return self.p_sample_loop(shape, condition=cond, timesteps=timesteps, verbose=verbose, x0=x0, x_T=x_T, return_intermediates=return_intermediates)

    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None, timesteps=None, verbose=True, x_T=None, return_intermediates=False, x0=None, start_T=None, mask=None):
        device = self.diffusion.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.diffusion.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling <t>', total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if mask is not None:  # TODO what is mask for
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, ts, condition=condition, clip_denoised=self.diffusion.clip_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % self.diffusion_logger_every == 0 or i == timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img


    @torch.no_grad()
    def p_sample(self, x, t, condition=None, clip_denoised=False, repeat_noise=False, return_x0=False, temperature=1., noise_dropout=0.):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x=x,
                                                                          c=condition,
                                                                          t=t,
                                                                          clip_denoised=clip_denoised
                                                                          )

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    def p_mean_variance(self, x, c, t,
                        clip_denoised: bool):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        based on models prediction of noise(aka eps), and given t, tries to predict the x_0, then computes the posterior of mu and var
        """

        model_output = self.model(x, t, c)

        if self.diffusion.parameterization == 'eps':
            x_recon = self.diffusion.predict_start_from_noise(x, t=t, noise=model_output)
        else:
            raise NotImplementedError('only epsilon prediction is implemented')
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @contextmanager
    def ema_scope(self, context=None):
        if self.with_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.with_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = DareDataset(root_path='C:\\Users\massoud\PycharmProjects\latent_ddpm\latent_ddpm\data', mode='train', transform=image_transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset = DareDataset(root_path='C:\\Users\massoud\PycharmProjects\latent_ddpm\latent_ddpm\data', mode='validation', transform=image_transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

